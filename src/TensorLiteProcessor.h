#ifndef TENSORLITEPROCESSOR_H
#define TENSORLITEPROCESSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <deque>
#include <thread>
#include <string>
#include <map>

//#include <memory>
//#include <thread>
#include <condition_variable>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "ReadClassesToLabels.h"
#include "MessageQueue.h"
#include "MessageQueue.cpp"

using namespace std;
using namespace cv;

class TensorLiteProcessorClass;
class DetectorLiteClass;
class DetectionResultClass;

struct Detection_t
{
  float score;
  int detclass;
  Point2d BoxTopLeft;
  Point2d BoxBottomRigth;
  string ClassName;
};

struct BoundingBox_t
{
  float y1;
  float x1;
  float y2;
  float x2;
};

typedef struct BoundingBoxUint8_t
{
  uint8_t top;
  uint8_t left;
  uint8_t bottom;
  uint8_t right;
} BoundingBoxUint8_t;

class DetectionResultClass
{

private:
  unique_ptr<Mat> _image;
  int _num_detections;
  mutex _mtx;

protected:
  vector<Detection_t> _detections;

public:
  const Mat &GetImage() { return *_image; }
  unique_ptr<Mat> MoveImage() { return move(_image); }
  const vector<Detection_t> &GetDetections() const { return _detections; }
  int runtime = 0;

  DetectionResultClass(unique_ptr<Mat> Image) : _image(move(Image)){};
  void AddDetection(Detection_t det) { _detections.emplace_back(det); }

  //move constructor
  DetectionResultClass(DetectionResultClass &&source)
  {
    _image = move(source._image);
    _num_detections = source._num_detections;
    _detections = source._detections;
    runtime = source.runtime;
    //do not move mutextes _mtx (move(source._mtx));
    source._image = nullptr;
  }

  //copy constructor
  DetectionResultClass(const DetectionResultClass &source)
  {
    _num_detections = source._num_detections;
    _detections = source._detections;
    runtime = source.runtime;
    //do not copy mutextes  _mtx = source._mtx;

    //alocate memory and copy image data
    if (source._image)
      _image = make_unique<Mat>(source._image->clone());
    else
      _image = nullptr;
  }

  //move assign
  DetectionResultClass &operator=(DetectionResultClass &&source)
  {
    if (this == &source)
      return *this;

    _image = move(source._image);
    runtime = source.runtime;
    _num_detections = source._num_detections;

    source._image = nullptr;
    return *this;
  }
  //copy assignment operator
  DetectionResultClass &operator=(DetectionResultClass &source)
  {
    runtime = source.runtime;
    _num_detections = source._num_detections;
    if (this == &source)
      return *this;

    if (_image)
      _image = nullptr;

    if (source._image)
      _image = make_unique<Mat>(source._image->clone());
    else
      _image = nullptr;
    return *this;
  }
};

// generic Detector Class
class DetectorLiteClass
{
protected:
  DetectorLiteClass(string PtoModel) : _pathToModel(PtoModel){};

public:
  DetectorLiteClass() = delete;
  ~DetectorLiteClass(){};

  virtual const string GetDetectorName() = 0;
  virtual void FeedInterpreter(tflite::Interpreter &Interpreter, const Mat &_image) = 0;
  virtual void ProcessResults(unique_ptr<tflite::Interpreter> &Interpreter, DetectionResultClass &SessionResult) = 0;

  const string _pathToModel;

  //vector<TensorDescription> &GetOutputTensorDescriptions() { return _outputTensorDescriptions; }
  //const TensorDescription &GetInputTensorDescription() { return _inputTensorDescription; }
  //const int GetImageWidth() { return _inputTensorDescription.Width; }
  //const int GetImageHeigth() { return _inputTensorDescription.Height; }

  //const int GetInputTensorSize() { return 1; }

  //const string tag;
  //const string signature;

protected:
  //vector<TensorDescription> _outputTensorDescriptions;
  //TensorDescription _inputTensorDescription;
};

//specific class for Mobilenet v1
// https://tfhub.dev/tensorflow/ssd_mobilenet_v1
class MobilenetV1Class : public DetectorLiteClass
{
public:
  MobilenetV1Class(string &PathToModel) : DetectorLiteClass(PathToModel)
  {
    string filename = _pathToModel + "/" + "mscoco_label_map.pbtxt";
    _detClasses = ReadClasses2Labels(filename);
  }

  ~MobilenetV1Class(){};

  void FeedInterpreter(tflite::Interpreter &interpreter, const Mat &image)
  {
    /*
    unsigned char *input = (unsigned char *)(_image.data);
    for (int j = 0; j < _image.rows; j++)
    {
      for (int i = 0; i < _image.cols; i++)
      {
        unsigned char b = input[_image.step * j + i];
        unsigned char g = input[_image.step * j + i + 1];
        unsigned char r = input[_image.step * j + i + 2];
      }
    }*/

    // Get Input and Output tensors info
    int in_id = interpreter.inputs()[0];
    TfLiteTensor *in_tensor = interpreter.tensor(in_id);
    uint8_t *input = in_tensor->data.uint8;

    //copy image into input tensor
    //uint8_t *input = interpreter.typed_input_tensor<uint8_t>(0);

    //cout << "Pointer to Input Tensor: " << *input << endl;
    //uint8_t *image = (uint8_t *)_image.data;

    //put tensor data in Mat object
    Mat resized(300, 300, CV_8UC3, in_tensor->data.uint8);

    //cout << "Resizing" << endl;
    resize(image, resized, Size(300, 300));
    //memcpy(input, resized.data, 300*300*sizeof(unsigned char));
  }

  const string GetDetectorName() override
  {
    return "MobilenetV1";
  }

  void ProcessResults(unique_ptr<tflite::Interpreter> &Interpreter, DetectionResultClass &SessionResult) override
  {
    //T *output = interpreter->typed_output_tensor<T>(i);

    //decode detection
    int num_detections = Interpreter->typed_output_tensor<float>(3)[0];
    cout << "Decoding " << num_detections << " detections." << endl;

    for (int i = 0; i < num_detections; i++)
    {
      Detection_t Detection;
      Detection.score = Interpreter->typed_output_tensor<float>(2)[i];
      Detection.detclass = (int)(Interpreter->typed_output_tensor<float>(1)[i]);

      //cast float into Bounding Box type
      float boxf = Interpreter->typed_output_tensor<float>(0)[i];
      BoundingBoxUint8_t *box = (BoundingBoxUint8_t *)&boxf;

      Detection.BoxTopLeft.x = (int)(box->left / 300.0 * SessionResult.GetImage().cols);
      Detection.BoxTopLeft.y = (int)(1.0 - (box->top / 300.0) * SessionResult.GetImage().rows);

      Detection.BoxBottomRigth.x = (int)(box->right / 300.0 * SessionResult.GetImage().cols);
      Detection.BoxBottomRigth.y = (int)(1.0 - (box->bottom / 300.0) * SessionResult.GetImage().rows);

      //Detection.ClassName = _detClasses.find(Detection.detclass)->second;

      Detection.ClassName = "top: " + to_string((int)box->top) + " left: " + to_string((int)box->left) + " bottom: " + to_string((int)box->bottom) + " right: " + to_string((int)box->right);

      cout << "Box ID " << i << "top: " << (int)box->top << " left: " << (int)box->left << " bottom: " << (int)box->bottom << " right: " << (int)box->right << endl;

      SessionResult.AddDetection(move(Detection));
    }
  }

private:
  map<int, string> _detClasses;
};

class TensorLiteProcessorClass
{
public:
  TensorLiteProcessorClass(shared_ptr<DetectorLiteClass> Detector);
  ~TensorLiteProcessorClass();

  void StartProcessorThread();
  void StopProcessorThread();

  MessageQueue<unique_ptr<Mat>> input_queue;
  MessageQueue<DetectionResultClass> output_queue;

  void SessionRunLoop();

private:
  shared_ptr<DetectorLiteClass> _detector;
  thread _detectorThread;

  std::unique_ptr<tflite::Interpreter> _interpreter;
};

#endif