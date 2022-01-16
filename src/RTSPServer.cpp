#include "RTSPServer.h"

#include <gstreamer-1.0/gst/gst.h>
#include <gstreamer-1.0/gst/rtsp-server/rtsp-server.h>

static gboolean timeout(GstRTSPServer *server)
{
  GstRTSPSessionPool *pool;

  pool = gst_rtsp_server_get_session_pool(server);
  gst_rtsp_session_pool_cleanup(pool);
  g_object_unref(pool);

  return TRUE;
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
//x264enc
  gst_rtsp_media_factory_set_launch(_factory, "( "
                                              "videotestsrc ! video/x-raw,width=352,height=288,framerate=15/1 ! "
                                              "omxh264enc ! rtph264pay name=pay0 pt=96 "
                                              "audiotestsrc ! audio/x-raw,rate=8000 ! "
                                              "alawenc ! rtppcmapay name=pay1 pt=97 "
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

  cout << "stream ready at rtsp://127.0.0.1:8554/test\n";
  g_main_loop_run(_loop);
}


typedef struct
{
  gboolean white;
  GstClockTime timestamp;
} MyContext;

/* called when we need to give data to appsrc */

static void need_data (GstElement * appsrc, guint unused, MyContext * ctx)
{
  GstBuffer *buffer;
  guint size;
  GstFlowReturn ret;

  size = 385 * 288 * 2;

  //block and wait for frame 

  buffer = gst_buffer_new_allocate (NULL, size, NULL);

  /* this makes the image black/white */
  gst_buffer_memset (buffer, 0, ctx->white ? 0xff : 0x0, size);

  ctx->white = !ctx->white;

  /* increment the timestamp every 1/2 second */
  GST_BUFFER_PTS (buffer) = ctx->timestamp;
  GST_BUFFER_DURATION (buffer) = gst_util_uint64_scale_int (1, GST_SECOND, 2);
  ctx->timestamp += GST_BUFFER_DURATION (buffer);

  g_signal_emit_by_name (appsrc, "push-buffer", buffer, &ret);
  gst_buffer_unref (buffer);
}

/* called when a new media pipeline is constructed. We can query the
 * pipeline and configure our appsrc */

static void media_configure (GstRTSPMediaFactory * factory, GstRTSPMedia * media, gpointer user_data)
{
  GstElement *element, *appsrc;
  MyContext *ctx;

  /* get the element used for providing the streams of the media */
  element = gst_rtsp_media_get_element (media);

  /* get our appsrc, we named it 'mysrc' with the name property */
  appsrc = gst_bin_get_by_name_recurse_up (GST_BIN (element), "mysrc");

  /* this instructs appsrc that we will be dealing with timed buffer */
  gst_util_set_object_arg (G_OBJECT (appsrc), "format", "time");
  /* configure the caps of the video */
  g_object_set (G_OBJECT (appsrc), "caps",
      gst_caps_new_simple ("video/x-raw",
          "format", G_TYPE_STRING, "RGB16",
          "width", G_TYPE_INT, 384,
          "height", G_TYPE_INT, 288,
          "framerate", GST_TYPE_FRACTION, 0, 1, NULL), NULL);

  ctx = g_new0 (MyContext, 1);
  ctx->white = FALSE;
  ctx->timestamp = 0;
  /* make sure ther datais freed when the media is gone */
  g_object_set_data_full (G_OBJECT (media), "my-extra-data", ctx,
      (GDestroyNotify) g_free);

  /* install the callback that will be called when a buffer is needed */
  g_signal_connect (appsrc, "need-data", (GCallback) need_data, ctx);
  gst_object_unref (appsrc);
  gst_object_unref (element);
}

void RTSPServerClass::StartRTSPServer()
{
  
  gst_init (NULL, NULL);

  _loop = g_main_loop_new (NULL, FALSE);

  /* create a server instance */
  _server = gst_rtsp_server_new ();

  /* get the mount points for this server, every server has a default object
   * that be used to map uri mount points to media factories */
  _mounts = gst_rtsp_server_get_mount_points (_server);

  /* make a media factory for a test stream. The default media factory can use
   * gst-launch syntax to create pipelines.
   * any launch line works as long as it contains elements named pay%d. Each
   * element with pay%d names will be a stream */
  _factory = gst_rtsp_media_factory_new ();
  gst_rtsp_media_factory_set_launch (_factory,
      "( appsrc name=mysrc ! videoconvert ! x264enc ! rtph264pay name=pay0 pt=96 )");

  /* notify when our media is ready, This is called whenever someone asks for
   * the media and a new pipeline with our appsrc is created */
  g_signal_connect (_factory, "media-configure", (GCallback) media_configure, NULL);

  /* attach the test factory to the /test url */
  gst_rtsp_mount_points_add_factory (_mounts, "/test", _factory);

  /* don't need the ref to the mounts anymore */
  g_object_unref (_mounts);

  /* attach the server to the default maincontext */
  gst_rtsp_server_attach (_server, NULL);

  /* start serving */
  cout << "stream ready at rtsp://127.0.0.1:8554/test\n";
  g_main_loop_run (_loop);

  return;
}