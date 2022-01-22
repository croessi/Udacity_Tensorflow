#ifndef RTSPSERVER_H_
#define RTSPSERVER_H_

#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>

#include <iostream>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <gstreamer-1.0/gst/gst.h>
#include <gstreamer-1.0/gst/rtsp-server/rtsp-server.h>

#include "MessageQueue.h"
#include "MessageQueue.cpp"

using namespace std;
using namespace cv;

typedef struct
{
    gboolean white;
    GstClockTime timestamp;
    
} MyContext;


class RTSPServerClass
{
private:
    GMainLoop *_loop;
    GstRTSPServer *_server;
    GstRTSPMountPoints *_mounts;
    GstRTSPMediaFactory *_factory;
    string _mountpoint;
    
    thread _RTSPServerThread;
    void StartRTSPServerDummy();
    void StartRTSPServer(MatSize size);
    static void RTSP_media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data);
    static void RTSP_need_data(GstElement *appsrc, guint unused, MyContext *ctx);


    static unique_ptr<Mat> _current_frame;
    
public:
    RTSPServerClass(string mountpoint) : _mountpoint(mountpoint){};
    ~RTSPServerClass(){};

    void StartRTSPServerThread(MatSize size);
    void StopRTSPServerThread ();

    static MessageQueue<unique_ptr<Mat>> input_queue;
        //static MatSize _size;
};
#endif