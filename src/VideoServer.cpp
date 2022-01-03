#include "VideoServer.h"

#include <iostream>

void VideoServerClass::StartVideoServerThread()
{
  _VideoServerThread = thread(&VideoServerClass::SessionRunLoop, this);
}
void VideoServerClass::StopVideoServerThread()
{
  //create empty frame and set it to queue to trigger thread exit
  unique_ptr<Mat> frame = make_unique<Mat>(Mat(Size(0, 0), CV_8U));

  input_queue.sendAndClear(move(frame));
  cout << "Waiting for Detector thread to join.\n";
  _VideoServerThread.join();
}

void VideoServerClass::SessionRunLoop()
{

  cout << "VideoServer thread started.\n";
  while (true)
  {
    //wait for empty detection to be received
    unique_ptr<Mat> frame(input_queue.receive());

    //cout << "Sender Queue Size: " << input_queue.GetSize() << "\n";

    //cout << "Try to write frame to " << _dest_IP << " with size: " << frame->size[0] << "...";
    //empty frame --> cleanup and exit thread
    if (frame->size[0] == 0)
      return;

    if (!_video.isOpened())
    {
      cout << "Try to open Frame Writer\n";
      //x264enc tune=zerolatency speed-preset=ultrafast
      //video.open("appsrc ! videoconvert ! x264enc me=hex tune=zerolatency speed-preset=ultrafast key-int-max=50 intra-refresh=true ! h264parse ! rtph264pay config-interval=5 pt=96 ! udpsink host=" + dest_IP+ " port=5000 sync=false",0, 20, frame.size(), true);
      int fourcc = VideoWriter::fourcc('H', '2', '6', '4');
      _video.open("appsrc ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast key-int-max=2 ! h264parse ! rtph264pay config-interval=5 pt=96 ! udpsink host=" + _dest_IP + " port=5000 sync=false", CAP_GSTREAMER, fourcc, 5, frame->size(), true);

      if (!_video.isOpened())
      {
        cout << "Frame Writer not availbale!!!!!!\n";
        return;
      }
      cout << "Frame Writer open\n";
    }

    _video.write(*frame);
    //cout << "Video written";
  }
}
