#include "RTSPServer.h"
#include <string>
#include <gstreamer-1.0/gst/gst.h>
#include <gstreamer-1.0/gst/rtsp-server/rtsp-server.h>

static int g_width = 0;
static int g_height = 0;

static gboolean timeout(GstRTSPServer *server)
{
  GstRTSPSessionPool *pool;

  pool = gst_rtsp_server_get_session_pool(server);
  gst_rtsp_session_pool_cleanup(pool);
  g_object_unref(pool);

  return TRUE;
}

void RTSPServerClass::StartRTSPServerThread(MatSize size)
{
  g_width = size[1];
  g_height = size[0] + size[0] % 16; // for omxh264enc size has to be mutiples of 16
  _RTSPServerThread = thread(&RTSPServerClass::StartRTSPServer, this, size);
  //_RTSPServerThread = thread(&RTSPServerClass::StartRTSPServerDummy, this);
}

void RTSPServerClass::StopRTSPServerThread()
{
  //create empty frame and set it to queue to trigger thread exit
  unique_ptr<Mat> frame = make_unique<Mat>(Mat(Size(0, 0), CV_8U));

  RTSPServerClass::input_queue.sendAndClear(move(frame));
  cout << "Waiting for RTSPServerThread thread to join.\n";
  _RTSPServerThread.join();
}

/* called when we need to give data to appsrc */

//statistics how often need dada is called
auto t1 = chrono::high_resolution_clock::now();

void RTSPServerClass::RTSP_need_data(GstElement *appsrc, guint unused, MyContext *ctx)
{
  //statistics how often need dada is called
  auto t2 = chrono::high_resolution_clock::now();
  auto ms_int = chrono::duration_cast<chrono::milliseconds>(t2 - t1);
  t1 = t2;

  avgframecycle = (avgframecycle + ms_int.count()) / 2;
  rtsp_frames++;

  GstBuffer *buffer;
  guint size;
  GstFlowReturn ret;

  if (input_queue.GetSize() > 0)
  {
    cout << "ReceiveFrame" << endl;
    _current_frame = RTSPServerClass::input_queue.receive();
    // cout << "-------------------------------------------received Frame with size: " << _current_frame->size[1] << "x" << _current_frame->size[0] << endl;
    //append rows if g_height is bigger due to macroblock fitting
    if (g_height != _current_frame->size[0])
      _current_frame->resize(g_height);
  }

  if (!_current_frame)
  {
    cout << "Have no frame yet --> Create empty frame!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    _current_frame = make_unique<Mat>(g_width, g_height, CV_8UC3);
  }

  size = g_width * g_height * 3;
  //put frame into buffer
  buffer = gst_buffer_new_wrapped_full((GstMemoryFlags)0, (gpointer)(_current_frame->data), size, 0, size, NULL, NULL);

  ctx->white = !ctx->white;

  /* increment the timestamp every 1/2 second */
  GST_BUFFER_PTS(buffer) = ctx->timestamp;
  GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, 2);
  ctx->timestamp += GST_BUFFER_DURATION(buffer);

   cout << "pushing buffer" << endl;
  g_signal_emit_by_name(appsrc, "push-buffer", buffer, &ret);
  gst_buffer_unref(buffer);
     cout << "done" << endl;
}

/* called when a new media pipeline is constructed. We can query the
 * pipeline and configure our appsrc */

void RTSPServerClass::RTSP_media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data)
{
  GstElement *element, *appsrc;
  MyContext *ctx;

  /* get the element used for providing the streams of the media */
  element = gst_rtsp_media_get_element(media);

  /* get our appsrc, we named it 'mysrc' with the name property */
  appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(element), "mysrc");

  /* this instructs appsrc that we will be dealing with timed buffer */
  gst_util_set_object_arg(G_OBJECT(appsrc), "format", "time");
  /* configure the caps of the video */

  cout << "\n\n----------------------Creating Source with size :" << g_width << "x" << g_height << "------------------------\n\n";
  g_object_set(G_OBJECT(appsrc), "caps",
               gst_caps_new_simple("video/x-raw",
                                   "format", G_TYPE_STRING, "BGR",
                                   "width", G_TYPE_INT, g_width,
                                   "height", G_TYPE_INT, g_height,
                                   "framerate", GST_TYPE_FRACTION, 30, 1, NULL), //0,5frames /sec
               NULL);

  ctx = g_new0(MyContext, 1);
  ctx->white = FALSE;
  ctx->timestamp = 0;
  /* make sure ther datais freed when the media is gone */
  g_object_set_data_full(G_OBJECT(media), "my-extra-data", ctx, (GDestroyNotify)g_free);

  /* install the callback that will be called when a buffer is needed */
  g_signal_connect(appsrc, "need-data", (GCallback)RTSPServerClass::RTSP_need_data, ctx);
  gst_object_unref(appsrc);
  gst_object_unref(element);
}

void RTSPServerClass::StartRTSPServer(MatSize size)
{
  gst_init(NULL, NULL);

  _loop = g_main_loop_new(NULL, FALSE);

  /* create a server instance */
  _server = gst_rtsp_server_new();

  /* get the mount points for this server, every server has a default object
   * that be used to map uri mount points to media factories */
  _mounts = gst_rtsp_server_get_mount_points(_server);

  /* make a media factory for a test stream. The default media factory can use
   * gst-launch syntax to create pipelines.
   * any launch line works as long as it contains elements named pay%d. Each
   * element with pay%d names will be a stream */
  _factory = gst_rtsp_media_factory_new();
  //gst_rtsp_media_factory_set_launch(_factory, "( appsrc name=mysrc ! videoconvert ! omxh264enc ! rtph264pay name=pay0 pt=96 )");

  //WORKING gst_rtsp_media_factory_set_launch(_factory, "( appsrc name=mysrc ! videoconvert ! x264enc ! rtph264pay name=pay0 pt=96 )");
  //WORLING gst_rtsp_media_factory_set_launch(_factory, "( appsrc name=mysrc ! videoconvert ! videorate ! omxh264enc ! video/x-h264,profile=baseline,framerate=15/1 ! rtph264pay name=pay0 pt=96 )");
  //gst_rtsp_media_factory_set_launch(_factory, "( appsrc name=mysrc ! videoconvert ! v4l2h264enc ! video/x-h264,profile=baseline,framerate=15/1 ! rtph264pay name=pay0 pt=96 )");

  string pipe = "( " + _pipe + " )";
  gst_rtsp_media_factory_set_launch(_factory, pipe.c_str());

  //gst_rtsp_media_factory_set_launch(_factory, "( appsrc name=mysrc ! video/x-raw,format=RGB,width=640,height=360,framerate=1/2 ! videoconvert ! omxh264enc ! rtph264pay name=pay0 pt=96 )");
  //gst_rtsp_media_factory_set_launch(_factory, "( appsrc name=mysrc ! video/x-raw,format=RGB,width=640,height=360,framerate=15/1 ! videoconvert ! x264enc ! rtph264pay name=pay0 pt=96 )");
  //gst_rtsp_media_factory_set_launch(_factory, "( appsrc name=mysrc ! video/x-raw,format=RGB,width=640,height=360,framerate=15/1 ! videoconvert ! omxh264enc ! rtph264pay name=pay0 pt=96 )");

  //! video/x-raw,width=352,height=288,framerate=15/1 ! omxh264enc ! rtph264pay name=pay0 pt=96 "

  /* notify when our media is ready, This is called whenever someone asks for
   * the media and a new pipeline with our appsrc is created */
  g_signal_connect(_factory, "media-configure", (GCallback)RTSPServerClass::RTSP_media_configure, NULL);
  //this->media_configure
  /* attach the test factory to the /test url */
  gst_rtsp_mount_points_add_factory(_mounts, "/DetectorPi", _factory);

  /* don't need the ref to the mounts anymore */
  g_object_unref(_mounts);

  /* attach the server to the default maincontext */
  gst_rtsp_server_attach(_server, NULL);

  /* start serving */
  cout << "stream ready at rtsp://127.0.0.1:8554/DetectorPi\n";
  //g_main_loop_run(_loop);
}

void RTSPServerClass::StartRTSPServerDummy()
{

  gst_init(NULL, NULL);

  _loop = g_main_loop_new(NULL, FALSE);

  /* create a server instance */
  _server = gst_rtsp_server_new();

  /* get the mount points for this server, every server has a default object
   * that be used to map uri mount points to media factories */
  _mounts = gst_rtsp_server_get_mount_points(_server);

  /* make a media factory for a test stream. The default media factory can use
   * gst-launch syntax to create pipelines.
   * any launch line works as long as it contains elements named pay%d. Each
   * element with pay%d names will be a stream */
  _factory = gst_rtsp_media_factory_new();
  //x264enc omxh264enc
  gst_rtsp_media_factory_set_launch(_factory, "( "
                                              "videotestsrc ! video/x-raw,format=RGB,width=352,height=288,framerate=15/1 ! videoconvert !"
                                              "omxh264enc ! rtph264pay name=pay0 pt=96 "
                                              //"audiotestsrc ! audio/x-raw,rate=8000 ! "
                                              //"alawenc ! rtppcmapay name=pay1 pt=97 "
                                              ")");

  /* attach the test factory to the /test url */
  gst_rtsp_mount_points_add_factory(_mounts, "/test", _factory);

  /* don't need the ref to the mapper anymore */
  g_object_unref(_mounts);

  /* attach the server to the default maincontext */
  if (gst_rtsp_server_attach(_server, NULL) == 0)
    cout << "------------------RTSP Server Failed to init!!! ------------------ \n";

  /* add a timeout for the session cleanup */
  g_timeout_add_seconds(2, (GSourceFunc)timeout, _server);

  /* start serving, this never stops */

  cout << "Teststream ready at rtsp://127.0.0.1:8554/test\n";
  //g_main_loop_run(_loop);
  //cout << "Return \n";
  //debug output pipe
  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(_factory), GST_DEBUG_GRAPH_SHOW_ALL, "outpupt_pipe_testsrc");
}