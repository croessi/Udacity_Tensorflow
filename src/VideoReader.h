
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
    const string _filename;
    unique_ptr<VideoCapture> _cap;
    vector<Mat> _frameBuffer;
    mutex _mut;

    bool _isokay;

public:
    VideoReader(string filename) : _filename(filename), _cap(nullptr){};
    ~VideoReader() { _cap.release(); };
    string getFilename() { return _filename; }

    //this thread allways tries to fill the framebuffe with 5 frames
    void FrameReadLoop();

    unique_ptr<Mat> getNextFrame();
};

#endif