#ifndef VIDEOSERVER_H
#define VIDEOSERVER_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <deque>
#include <thread>
#include <string>
#include <map>

#include <condition_variable>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "MessageQueue.h"
#include "MessageQueue.cpp"

using namespace std;
using namespace cv;

// generic Detector Class
class VideoServerClass
{
public:
  VideoServerClass(string dest_IP) : _dest_IP(dest_IP){};

  ~VideoServerClass(){
    if (_video.isOpened())
      _video.release();
}

  void StartVideoServerThread();
  void StopVideoServerThread();

  void SessionRunLoop();

  const string _dest_IP;
  MessageQueue<unique_ptr<Mat>> input_queue;

  private:
  VideoWriter _video;

  thread _VideoServerThread;

};

#endif