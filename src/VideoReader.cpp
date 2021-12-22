#include "VideoReader.h"

#include <iostream>

void VideoReader::FrameReadLoop()
{
    unique_lock<mutex> lck(_mut);
    //create cpature instance
    if (!_cap)
        _cap = make_unique<VideoCapture>(_filename);

    while (true)
    {

        //only cpature mor frames if buffer runs low
        if (_frameBuffer.size() < 5)
        {
            lck.unlock();

            Mat f;
            *_cap >> f;

            /// If the frame is empty,errormessage
            if (f.empty())
            {
                cout << "No more frames -> exit framegrabber thread."; 
                return;
            }
        
            lck.lock();
            _frameBuffer.emplace(_frameBuffer.begin(), move(f));
        }

        lck.unlock();

        //sleep
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        lck.lock();
    }
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
