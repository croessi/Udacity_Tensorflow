
#ifndef VIDEOREADER_H_
#define VIDEOREADER_H_

#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

class VideoReader
{
private:
    const string _input;
    unique_ptr<VideoCapture> _cap;
    vector<Mat> _frameBuffer;
    mutex _mut;
    float _scalingFactor;
    thread _readFrameLoopThread;

    bool _exitThread;

public:
    VideoReader(string input, float scalingFactor = 1.0) : _input(input), _scalingFactor(scalingFactor), _cap(nullptr), _exitThread(false){};
    ~VideoReader() { _cap.release(); };

    void StartGrabberThread();
    void StopGrabberThread();

    string getInput() { return _input; }

    //this thread allways tries to fill the framebuffe with 5 frames
    void FrameReadLoop();

    unique_ptr<Mat> getNextFrame();
};

#endif