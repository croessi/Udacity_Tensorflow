#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>
#include <vector>
#include <future>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "ReadClassesToLabels.h"

using namespace cv;
using namespace std;

void NoOpDeallocator(void *data, size_t a, void *b) {}

struct Detection
{
  float score;
  int detclass;
};

struct DetectionResult
{
  unique_ptr<Mat> Image;
  int num_detections;
  vector<Detection> Detections;
};

class TensorProcessor
{
private:
public:
  TensorProcessor();
  ~TensorProcessor();

  bool NewDetectionAvailable;
};

class VideoReader
{
private:
  const string _filename;
  unique_ptr<VideoCapture> _cap;
  promise<void> _prmsFrameRead;
  future<void> _ftrFrameRead;
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
  //load labels
  map<int, string> DetClasses = ReadClasses2Labels("../mscoco_label_map.pbtxt");

  /*
  VideoCapture cap("../output.mp4");
  if (!cap.isOpened())
  {

    cout << "Error opening chaplin video stream or file" << endl;

    return -1;
  }

  //read frame
  Mat f;
  cap >> f;
*/

  //create instance of video reader
  VideoReader Reader("/home/vscode/Udacity_Tensorflow/output.mp4");

  promise<void> prmsVideoOpen;
  future<void> ftrVideoOpen = prmsVideoOpen.get_future();

  thread readFrameLoopThread(&VideoReader::FrameReadLoop, &Reader);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  //wait until vide is open
  //ftrVideoOpen.get();

  //get first frame
  unique_ptr<Mat> frame(Reader.getNextFrame());

  frame = Reader.getNextFrame();

  //Load CNN
  //********* Read model
  TF_Graph *Graph = TF_NewGraph();
  TF_Status *Status = TF_NewStatus();

  TF_SessionOptions *SessionOpts = TF_NewSessionOptions();
  TF_Buffer *RunOpts = NULL;

  const char *saved_model_dir = "../ssd_mobilenet_v2"; // Path of the model
  const char *tags = "serve";                          // default model serving tag; can change in future
  int ntags = 1;

  TF_Session *Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
  if (TF_GetCode(Status) == TF_OK)
  {
    printf("TF_LoadSessionFromSavedModel OK\n");
  }
  else
  {
    printf("%s", TF_Message(Status));
  }

  //****** Get input tensor
  int NumInputs = 1;
  TF_Output *Input = (TF_Output *)malloc(sizeof(TF_Output) * NumInputs);

  TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_tensor"), 0};
  if (t0.oper == NULL)
    printf("ERROR: Failed TF_GraphOperationByName serving_default_input_tensor\n");
  else
    printf("Input Tensor created and assigned\n");

  Input[0] = t0;

  //********* Get Output tensor
  int NumOutputs = 8;
  TF_Output *Output = (TF_Output *)malloc(sizeof(TF_Output) * NumOutputs);

  for (int i = 0; i < NumOutputs; i++)
  {
    TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), i};
    if (t2.oper == NULL)
      printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");

    Output[i] = t2;
  }

  printf("Output Tensors created and assigned\n");

  //********* Allocate data for inputs & outputs
  TF_Tensor **InputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * NumInputs);
  TF_Tensor **OutputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * NumOutputs);

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

  int ndims = 4;

  int64_t dims[] = {1, frame->size[0], frame->size[1], 3};
  int ndata = dims[0] * dims[1] * dims[2] * dims[3];

  while (true)
  {

    //unique_ptr<TF_Tensor> int_tensor = make_unique<TF_Tensor>(TF_NewTensor(TF_UINT8, dims, ndims, frame.data, ndata, &NoOpDeallocator, 0));

    TF_Tensor *int_tensor = TF_NewTensor(TF_UINT8, dims, ndims, frame->data, ndata, &NoOpDeallocator, 0);

    if (int_tensor != NULL)
      printf("TF_NewTensor is OK\n");
    else
      printf("ERROR: Failed TF_NewTensor\n");

    InputValues[0] = int_tensor;

    //Run the Session
    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL, Status);

    if (TF_GetCode(Status) == TF_OK)
      printf("Session rn through\n");
    else
      printf("%s", TF_Message(Status));

    //get num of detections
    int num_detections = *(float *)(TF_TensorData(OutputValues[5]));

    //display result
    for (int i = 0; i < num_detections; i++)
    {
      float score = ((float *)TF_TensorData(OutputValues[4]))[i];
      int detclass = ((float *)TF_TensorData(OutputValues[2]))[i];

      cout << "Detection Score " << i << ": " << score << " " << DetClasses[detclass] << "\n";
    }

    //get next frame
    frame = Reader.getNextFrame();
  }

  // When everything done, release the video capture object
  destroyAllWindows();

  // //Free memory
  TF_DeleteGraph(Graph);
  TF_DeleteSession(Session, Status);
  TF_DeleteSessionOptions(SessionOpts);
  TF_DeleteStatus(Status);

  return 0;
}
