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
    unique_lock<mutex> lck(_mut);
    //create cpature instance
    
    const std::string RTSP_URL = "rtsp://192.168.0.49:554/ch0_0.h264";
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp", 1);
    
    VideoWriter video;
    video.open("appsrc ! autovideoconvert ! omxh265enc ! matroskamux ! filesink location=test.mkv sync=false", 0, (double)20, cv::Size(1920/2, 1080/2), true);

    if (!video.isOpened()) {
        printf("can't create writer\n");
    return;
    }

    if (!_cap){
        //_cap = make_unique<VideoCapture>(_filename);
        //_cap =  make_unique<VideoCapture>("rtsp://192.168.0.49/ch0_0.h264",CAP_FFMPEG  );
        _cap =  make_unique<VideoCapture>(RTSP_URL, CAP_FFMPEG);
        if (!_cap->isOpened()) {
          std::cout << "Cannot open RTSP stream" << std::endl;
        return;
        }
    }
    while (true)
    {

        //only cpature mor frames if buffer runs low
        if (_frameBuffer.size() < 3)
        {
            lck.unlock();

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
                return;
            }

            if (!f.empty()){

                //rescale
                float scale_percent = 50;
                int d_width = f.size[1] * scale_percent / 100;
                int d_height = f.size[0] * scale_percent / 100;

                Mat f_small;
                resize (f,f_small,Size(d_width,d_height));
                _frameBuffer.emplace(_frameBuffer.begin(), move(f_small));
                f.release();

                //write video to file
                  video.write(f_small);
               
            }
            else
            {
            cout << "Empty frame & Framebuffer @" << _frameBuffer.size() << "frames\n";
            _cap->release ();
            _cap.release ();
            //_cap =  make_unique<VideoCapture>(Cap_pipeline,CAP_GSTREAMER);
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
