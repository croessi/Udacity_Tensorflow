#ifndef GSTREAMER_H_
#define GSTREAMER_H_

#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>

#include <iostream>
#include <memory>

#include <gstreamer-1.0/gst/gst.h>
#include <gstreamer-1.0/gst/rtsp-server/rtsp-server.h>

using namespace std;

class RTSPServerClass
{
private:
    GMainLoop* _loop;
    GstRTSPServer* _server;
    GstRTSPMountPoints* _mounts;
    GstRTSPMediaFactory* _factory;
    string _mountpoint;

public:
    RTSPServerClass(string mountpoint) : _mountpoint(mountpoint){};
    ~RTSPServerClass(){};

    void StartRTSPServerDummy();
        void StartRTSPServer();
};
#endif