#ifndef TENSORPOCESSOR_H
#define TENSORPOCESSOR_H

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
#include <tensorflow/c/c_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "ReadClassesToLabels.h"

using namespace std;
using namespace cv;

class TensorProcessorClass;
class DetectorClass;
class DetectionResultclass;

// generic Detector Class
class DetectorClass
{
protected:
  DetectorClass(string nameOfInputs, int numInputs, string nameOfOutputs, int numOutputs, string PtoModel, map<int, string> detClasses) : _nameOfInputs(nameOfInputs),
                                                                                                                                          _numInputs(numInputs),
                                                                                                                                          _nameOfOutputs(nameOfOutputs),
                                                                                                                                          _numOutputs(numOutputs),
                                                                                                                                          _pathToModel(PtoModel),
                                                                                                                                          _detClasses(detClasses){};

public:
  DetectorClass() : _nameOfInputs(""),
                    _numInputs(1),
                    _nameOfOutputs(""),
                    _numOutputs(1),
                    _pathToModel(""),
                    _detClasses(){};
  ~DetectorClass(){};

  virtual string GetDetectorName() { return ""; }
  const string &GetStringFromClass(int classID) const { return _detClasses.find(classID)->second; }
  const map<int, string> &GetDetClasses() const { return _detClasses; }

  const int _numInputs;
  const int _numOutputs;
  const string _nameOfOutputs;
  const string _nameOfInputs;
  const string _pathToModel;

private:
  const map<int, string> _detClasses;
};

//specific class for Mobilenet containing its config
// https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2
class MobilenetV2Class : public DetectorClass
{
public:
  MobilenetV2Class(string PathToModel) : DetectorClass("serving_default_input_tensor", 1, "StatefulPartitionedCall", 8, PathToModel, ReadClasses2Labels(PathToModel + "/" + "mscoco_label_map.pbtxt")){};
  ~MobilenetV2Class(){};

  string GetDetectorName() override { return "MobilenetV2Class"; }
};

struct Detection_t
{
  float score;
  int detclass;
  Point2d BoxTopLeft;
  Point2d BoxBottomRigth;
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

  friend TensorProcessorClass; //make it friend to allow acess from Processor Class to Detections (but not other classes)

public:
  const Mat &GetImage() { return *_image; }
  const vector<Detection_t> &GetDetections() const { return _detections; }

  DetectionResultClass(unique_ptr<Mat> Image) : _image(move(Image)){};

  //move constructor
  DetectionResultClass(DetectionResultClass &&source)
  {
    _image = move(source._image);
    _num_detections = source._num_detections;
    _detections = source._detections;
    //do not move mutextes _mtx (move(source._mtx));
    source._image = nullptr;
  }

  //copy constructor
  DetectionResultClass(const DetectionResultClass &source)
  {
    _num_detections = source._num_detections;
    _detections = source._detections;
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

    if (_image)
      _image = nullptr;

    _image = move(source._image);
    source._image = nullptr;
    return *this;
  }
  //copy assignment operator
  DetectionResultClass &operator=(DetectionResultClass &source)
  {
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

  //todo move constructores etc.
  bool DetectionDone = false;
};

template <class T>
class MessageQueue
{
public:
  T receive()
  {
    // lock
    std::unique_lock<std::mutex> uLock(_mutex);
    _cond.wait(uLock, [this]
               { return !_messages.empty(); });

    // remove last element
    T message = std::move(_messages.back());
    _messages.pop_back();

    return message;
  }

  void send(T &&message)
  {
    // lock
    std::lock_guard<std::mutex> uLock(_mutex);

    // add vector to queue
    _messages.emplace_back(std::move(message));
    _cond.notify_one();
  }

  int GetSize()
  {
    // lock
    std::lock_guard<std::mutex> uLock(_mutex);
    return _messages.size();
  }

private:
  std::mutex _mutex;
  std::condition_variable _cond;
  std::deque<T> _messages;
};

class TensorProcessorClass
{
public:
  TensorProcessorClass(shared_ptr<DetectorClass> Detector);
  ~TensorProcessorClass();

  void StartProcessorThread();
  void StopProcessorThread();

  MessageQueue<DetectionResultClass> input_queue;
  MessageQueue<DetectionResultClass> output_queue;

  void SessionRunLoop();

private:
  TF_Graph *_graph;
  TF_SessionOptions *_sessionOpts;
  TF_Buffer *_runOpts;
  TF_Session *_session;
  TF_Status *_status;
  TF_Output *_input;
  TF_Output *_output;
  shared_ptr<DetectorClass> _detector;
  thread _detectorThread;

};

#endif