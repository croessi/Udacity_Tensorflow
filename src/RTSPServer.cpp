#include "RTSPServer.h"

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
  //_size = size;

  g_width = size[1];
  g_height = size[0];


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

void RTSPServerClass::RTSP_need_data(GstElement *appsrc, guint unused, MyContext *ctx)
{

  GstBuffer *buffer;
  guint size;
  GstFlowReturn ret;

  // cout << "---------------Need Data--------------------";
 
  
  /*
  size = g_width * g_height * 3;
  unsigned char tempframe[size];

  for (int i =0; i<size; i)
  {
    tempframe[i++]=0;
    tempframe[i++]=0;
    tempframe[i++]=255;
  }*/

  if (input_queue.GetSize() > 0)
  {
    _current_frame = RTSPServerClass::input_queue.receive();
    cout << "-------------------------------------------received Frame with size: " << _current_frame->size[1] << "x" << _current_frame->size[0] << endl;
  }

  if (!_current_frame)
  {
    cout << "Have no frame yet --> Create empty frame!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    _current_frame = make_unique<Mat>(g_width, g_height, CV_8UC3);
  }


  buffer = gst_buffer_new_wrapped_full((GstMemoryFlags)0, (gpointer)(_current_frame->data), size, 0, size, NULL, NULL);


 //buffer = gst_buffer_new_allocate (NULL, size, NULL);

  /* this makes the image black/white */
  //gst_buffer_memset (buffer, 0, ctx->white ? 0xff : 0x0, size);

  ctx->white = !ctx->white;

  /* increment the timestamp every 1/2 second */
  GST_BUFFER_PTS (buffer) = ctx->timestamp;
  GST_BUFFER_DURATION (buffer) = gst_util_uint64_scale_int (1, GST_SECOND, 2);
  ctx->timestamp += GST_BUFFER_DURATION (buffer);

  g_signal_emit_by_name (appsrc, "push-buffer", buffer, &ret);
  gst_buffer_unref (buffer);

  /*

  //get frame if availbale


  if (input_queue.GetSize() > 0)
  {
    _current_frame = RTSPServerClass::input_queue.receive();
    //  unique_ptr<Mat> frame(RTSPServerClass::input_queue.receive());
    //  _current_frame = make_unique<Mat>(_size[0],_size[1],CV_8UC3);
    //  cvtColor(*frame,*_current_frame,COLOR_RGB2YUV);

    //_current_frame = RTSPServerClass::input_queue.receive();
    cout << "-------------------------------------------received Frame with size: " << _current_frame->size[1] << "x" << _current_frame->size[0] << endl;
  }

  if (!_current_frame)
  {
    cout << "Have no frame yet --> Create empty frame!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    _current_frame = make_unique<Mat>(g_width, g_height, CV_8UC3);
  }

  //_current_frame->setTo(cv::Scalar(125,50,50));
  //size = _current_frame->size[0] * _current_frame->size[1] * 3;

  size = g_width * g_height * 3;

  // buffer = gst_buffer_new_wrapped_full((GstMemoryFlags)0, (gpointer)(_current_frame->data), size, 0, size, NULL, NULL);
  //gst_buffer_new();
  //  gst_buffer_set_data(buffer, _current_frame->data, _size[0]*_size[1]);

  //GST_BUFFER_DATA(buffer);

  //size = RTSP_size[0]*  RTSP_size[1] *3;
  //size = 384 * 288 * 2;

  buffer = gst_buffer_new_allocate(NULL, size, NULL);

  // this makes the image black/white
  gst_buffer_memset(buffer, 0, ctx->white ? 0xff : 0x0, size);

  ctx->white = !ctx->white;

  // increment the timestamp every 1/2 second
  GST_BUFFER_PTS(buffer) = ctx->timestamp;
  GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, 2);
  ctx->timestamp += GST_BUFFER_DURATION(buffer);

  g_signal_emit_by_name(appsrc, "push-buffer", buffer, &ret);
  gst_buffer_unref(buffer);

  //debug output pipe
  //GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(appsrc), GST_DEBUG_GRAPH_SHOW_ALL, "outpupt_pipe_2");

  //g_signal_emit_by_name(appsrc, "push-buffer", buffer, &ret);
  //g_signal_emit_by_name(appsrc, "push-buffer", _current_frame->data, &ret);
  // gst_buffer_unref(buffer);

  */

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

  cout << "----------------------Creating Source with size :" << g_width << "x" << g_height << endl;
  g_object_set(G_OBJECT(appsrc), "caps",
               gst_caps_new_simple("video/x-raw",
                                   "format", G_TYPE_STRING, "RGB",
                                   "width", G_TYPE_INT, g_width,
                                   "height", G_TYPE_INT, g_height,
                                   "framerate", GST_TYPE_FRACTION, 1, 2, NULL), //0,5frames /sec
               NULL);

  /*
  g_object_set(G_OBJECT(appsrc), "caps",
               gst_caps_new_simple("video/x-raw",
                                   "format", G_TYPE_STRING, "RGB16",
                                   "width", G_TYPE_INT, 384,
                                   "height", G_TYPE_INT, 288,
                                   "framerate", GST_TYPE_FRACTION, 1, 2, NULL), //0,5frames /sec
               NULL);
*/
  ctx = g_new0 (MyContext, 1);
  ctx->white = FALSE;
  ctx->timestamp = 0;
  /* make sure ther datais freed when the media is gone */
  g_object_set_data_full (G_OBJECT (media), "my-extra-data", ctx,
      (GDestroyNotify) g_free);

  /* install the callback that will be called when a buffer is needed */
  g_signal_connect (appsrc, "need-data", (GCallback) RTSPServerClass::RTSP_need_data, ctx);
  gst_object_unref (appsrc);
  gst_object_unref (element);
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
  gst_rtsp_media_factory_set_launch(_factory, "( appsrc name=mysrc ! videoconvert ! x264enc ! rtph264pay name=pay0 pt=96 )");
  //gst_rtsp_media_factory_set_launch(_factory, "( appsrc name=mysrc ! video/x-raw,format=RGB,width=640,height=360,framerate=1/2 ! videoconvert ! omxh264enc ! rtph264pay name=pay0 pt=96 )");
  //gst_rtsp_media_factory_set_launch(_factory, "( appsrc name=mysrc ! video/x-raw,format=RGB,width=640,height=360,framerate=15/1 ! videoconvert ! x264enc ! rtph264pay name=pay0 pt=96 )");
  //gst_rtsp_media_factory_set_launch(_factory, "( appsrc name=mysrc ! video/x-raw,format=RGB,width=640,height=360,framerate=15/1 ! videoconvert ! omxh264enc ! rtph264pay name=pay0 pt=96 )");

  //! video/x-raw,width=352,height=288,framerate=15/1 ! omxh264enc ! rtph264pay name=pay0 pt=96 "

  /* notify when our media is ready, This is called whenever someone asks for
   * the media and a new pipeline with our appsrc is created */
  g_signal_connect(_factory, "media-configure", (GCallback)RTSPServerClass::RTSP_media_configure, NULL);
  //this->media_configure
  /* attach the test factory to the /test url */
  gst_rtsp_mount_points_add_factory(_mounts, "/test", _factory);

  /* don't need the ref to the mounts anymore */
  g_object_unref(_mounts);

  /* attach the server to the default maincontext */
  gst_rtsp_server_attach(_server, NULL);

  /* start serving */
  cout << "stream ready at rtsp://127.0.0.1:8554/test\n";
  //g_main_loop_run(_loop);
  //cout << "Main Loop Started\n";

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