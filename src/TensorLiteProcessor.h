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
^
    // generic Detector Class
    class DetectorLiteClass
{
protected:
  DetectorLiteClass(string PtoModel) : _pathToModel(PtoModel){};

public:
  DetectorClass() = delete;
  ~DetectorClass(){};

  virtual const string GetDetectorName() = 0;
  virtual void SetImageSize(const int w, const int h) = 0;
  virtual void FeedInterpreter(unique_ptr<tflite::Interpreter> &Interpreter, const Mat &_image) = 0;
  virtual void ProcessResults(DetectionResultClass &SessionResult, TF_Tensor **OutputValues, int width, int heigth) = 0;

  const string _pathToModel;

  vector<TensorDescription> &GetOutputTensorDescriptions() { return _outputTensorDescriptions; }
  const TensorDescription &GetInputTensorDescription() { return _inputTensorDescription; }
  const int GetImageWidth() { return _inputTensorDescription.Width; }
  const int GetImageHeigth() { return _inputTensorDescription.Height; }

  const int GetInputTensorSize() { return 1; }

  const string tag;
  const string signature;

protected:
  vector<TensorDescription> _outputTensorDescriptions;
  TensorDescription _inputTensorDescription;
};

//specific class for Mobilenet v1
// https://tfhub.dev/tensorflow/ssd_mobilenet_v1
class MobilenetV1Class : public DetectorLiteClass
{
public:
  MobilenetV2Class(string &PathToModel) : DetectorClass(PathToModel, "serve", "serving_default", TensorDescription("serving_default_input_tensor", TF_UINT8, 0, {1, -1, -1, 3}))
  {
    string filename = _pathToModel + "/" + "mscoco_label_map.pbtxt";

    _detClasses = ReadClasses2Labels(filename);
  }
  ~MobilenetV2Class(){};

  void FeedInterpreter(unique_ptr<tflite::Interpreter> &Interpreter, const Mat &_image)
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

    //copy image into input tensor
    uint8_t *input = interpreter->typed_input_tensor<uint8_t>(0);
    uint8_t *image = (uint8_t *)_image.data;

    for (int i = 0; i < _image.rows * _image.cols; i++)
    {
      input[i] = *image[i];
    }
  }

  void SetImageSize(const int w, const int h)
  {
    _inputTensorDescription.Width = w;
    _inputTensorDescription.Height = h;
    _inputTensorDescription.dims = {1, h, w, 3};
  }

  const string GetDetectorName() override
  {
    return "MobilenetV2Class";
  }

  unique_ptr<Mat> ConvertImage(const Mat &OpenCVImage) override
  {
    Mat cp;
    OpenCVImage.copyTo(cp);
    return (make_unique<Mat>(move(cp)));
  }

  void ProcessResults(DetectionResultClass &SessionResult, TF_Tensor **OutputValues, int width, int heigth) override
  {
    //decode detection
    int num_detections = *(float *)(TF_TensorData(OutputValues[5]));
    for (int i = 0; i < num_detections; i++)
    {
      Detection_t Detection;
      Detection.score = ((float *)TF_TensorData(OutputValues[4]))[i];
      Detection.detclass = ((float *)TF_TensorData(OutputValues[2]))[i];

      BoundingBox_t box = ((BoundingBox_t *)TF_TensorData(OutputValues[1]))[i];

      Detection.BoxTopLeft.x = (int)(box.x1 * width);
      Detection.BoxTopLeft.y = (int)(box.y1 * heigth);

      Detection.BoxBottomRigth.x = (int)(box.x2 * width);
      Detection.BoxBottomRigth.y = (int)(box.y2 * heigth);
      Detection.ClassName = _detClasses.find(Detection.detclass)->second;

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