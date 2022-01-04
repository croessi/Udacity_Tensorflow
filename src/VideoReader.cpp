#include "VideoReader.h"

#include <iostream>

void VideoReader::StartGrabberThread()
{
    _readFrameLoopThread = thread(&VideoReader::FrameReadLoop, this);
}

void VideoReader::StopGrabberThread()
{
    unique_lock<mutex> lck(_mut);
    _exitThread = true;
    lck.unlock();
    cout << "Waiting for Frame Grabber thread to join.\n";
    _readFrameLoopThread.join();
}

void VideoReader::FrameReadLoop()
{

    cout << "Frame Grabber thread started.\n";
    unique_lock<mutex> lck(_mut);
    //create cpature instance

    if (!_cap)
        _cap = make_unique<VideoCapture>(_input, CAP_GSTREAMER);

    while (!_cap->isOpened())
    {
        std::cout << "Could not open RTSP stream to: " << _input << " will try again." << std::endl;
        _cap->open(_input, CAP_GSTREAMER);
        waitKey(10000);
    }

    while (true)
    {

        //only cpature more frames if buffer runs low
        if (_frameBuffer.size() < 2)
        {
            lck.unlock();

            //Mat f(Size(10000, 10000), CV_16F);
            Mat f;
            *_cap >> f;

            lck.lock();
            // If the frame is empty no more to read -> exit thread
            //if (f.empty() || _exitThread)
            if (_exitThread)
            {
                cout << "Exit Thread called -> exit framegrabber thread.\n";
                _frameBuffer.emplace(_frameBuffer.begin(), move(f));
                lck.unlock();
                _cap->release();
                _cap.release();
                return;
            }

            if (!f.empty())
            {

                //rescale
                if (_scalingFactor != 1.0)
                {
                    int d_width = f.size[1] * _scalingFactor;
                    int d_height = f.size[0] * _scalingFactor;

                    Mat f_small;
                    resize(f, f_small, Size(d_width, d_height));
                    f.release();
                    f = f_small;
                }

                _frameBuffer.emplace(_frameBuffer.begin(), move(f));
            }
            else
            {
                cout << "Empty frame & Framebuffer @" << _frameBuffer.size() << " frames. Try to reopen capture.\n";
                _cap->open(_input, CAP_GSTREAMER);
                waitKey(10000);
            }
        }

        lck.unlock();

        //sleep
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        lck.lock();
    }

    //Release video capture
    _cap->release();
}

unique_ptr<Mat> VideoReader::getNextFrame()
{
    while (true)
    {
        unique_lock<mutex> lck(_mut);

        if (_frameBuffer.size() > 0)
        {
            //get last element on framebuffer
            unique_ptr<Mat> frame = make_unique<Mat>(move(_frameBuffer.back()));
            _frameBuffer.pop_back();
            lck.unlock();
            return move(frame);
        }

        if (lck.owns_lock())
            lck.unlock();
        //wait for 1 msec before look again into the buffer
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    //return empty MAT object
    return {};
};
