#include <stdio.h>
#include <stdlib.h>
/*
#include <tensorflow/c/c_api.h>
#include <vector>
#include <future>

#include <deque>
*/

#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "TensorProcessor.h"

using namespace cv;
using namespace std;

class VideoReader
{
private:
  const string _filename;
  unique_ptr<VideoCapture> _cap;
  //promise<void> _prmsFrameRead;
  //future<void> _ftrFrameRead;
  vector<Mat> _frameBuffer;
  mutex _mut;

  bool _isokay;

public:
  //this thread allways tries to fill the framebuffe with 5 frames
  void FrameReadLoop()
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
          cout << "Error while reading Frame";

        lck.lock();
        _frameBuffer.emplace(_frameBuffer.begin(), move(f));
      }

      lck.unlock();

      //sleep
      std::this_thread::sleep_for(std::chrono::milliseconds(1));

      lck.lock();
    }
  }

public:
  VideoReader(string filename) : _filename(filename), _cap(nullptr){};
  ~VideoReader() { _cap.release(); };

  string getFilename();

  unique_ptr<Mat> getNextFrame()
  {
    /*
    //wait for the capture thread to return the image ( if thread already runing)
    if (_ftrFrameRead.valid())
      _ftrFrameRead.wait();

    //create new future
    _ftrFrameRead = _prmsFrameRead.get_future();

    //start new asynchronous thread to get next frame
    thread readNextF(&VideoReader::readNextFrame, this);

    //thread has alread put s.th to buffer
    */

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

  bool isOkay() { return _isokay; }
};

int main()
{

  //create instance of video reader
  VideoReader Reader("/home/vscode/Udacity_Tensorflow/output.mp4");

  //promise<void> prmsVideoOpen;
  //future<void> ftrVideoOpen = prmsVideoOpen.get_future();

  thread readFrameLoopThread(&VideoReader::FrameReadLoop, &Reader);
  //give thread som tme to instanciate
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  //load Model
  TensorProcessorClass TensorProcessor("../ssd_mobilenet_v2", 1, "serving_default_input_tensor", 8, "StatefulPartitionedCall");

  //start detector thread
  thread detectorThread(&TensorProcessorClass::SessionRunLoop, &TensorProcessor);
  //give thread som tme to instanciate
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  //unique_ptr<DetectionResultClass> DetectionResult = make_unique<DetectionResultClass>(Reader.getNextFrame());

  bool init = true;
  int c1 = 0;

  while (true)
  {
    unique_ptr<Mat> frame ( move(Reader.getNextFrame()));
    //check if we have new images or if we are in the first run
    if (TensorProcessor.output_queue.GetSize() > 0 || init)
    {

      //move frame into detection result
      DetectionResultClass SessionInput(move(frame));

      cout << "Frame Number" << c1++ << " is send to Detector \n";
      //move detection result into que to be processed by tensor flow
      TensorProcessor.input_queue.send(move(SessionInput));

      //skip in init run
      if (!init)
      {
        //get Image from detection and display
        DetectionResultClass SessionOutput(TensorProcessor.output_queue.receive());
        cout << "Detection Score of id: " << SessionOutput.GetDetections()[0].score << " " << TensorProcessor.GetStringFromClass(SessionOutput.GetDetections()[0].detclass) << "\n";
      }

      init = false;
    }
    else
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      cout << "Frame Number " << c1++ << " is displayed \n";
    }
  }

  /*
  std::string image_path = samples::findFile("../bus.jpg");
  Mat img_o = imread(image_path, IMREAD_COLOR);
  int scaling_factor = 7;
  Mat img; //(Size(img_o.size[0] / scaling_factor, img_o.size[1] / scaling_factor), CV_8U);
  resize(img_o, img, Size(img_o.size[0] / scaling_factor, img_o.size[1] / scaling_factor));

  if (img.empty())
  {
    std::cout << "Could not read the image: " << image_path << std::endl;
    return 1;
  }
*/
  //std::string image_path = samples::findFile("../bus.jpg");
  //Mat img_o = imread(image_path, IMREAD_COLOR);
  //int scaling_factor = 7;

  //Mat img; //(Size(img_o.size[0] / scaling_factor, img_o.size[1] / scaling_factor), CV_8U);
  //resize(img_o, img, Size(img_o.size[0] / scaling_factor, img_o.size[1] / scaling_factor));

  //get first frame
  //unique_ptr<Mat> frame (Reader.getNextFrame());

  //get first frame
  unique_ptr<Mat> frame(Reader.getNextFrame());

  while (true)
  {

    //unique_ptr<TF_Tensor> int_tensor = make_unique<TF_Tensor>(TF_NewTensor(TF_UINT8, dims, ndims, frame.data, ndata, &NoOpDeallocator, 0));

    //get next frame
    frame = Reader.getNextFrame();
  }

  // When everything done, release the video capture object
  destroyAllWindows();

  // //Free memory

  return 0;
}
