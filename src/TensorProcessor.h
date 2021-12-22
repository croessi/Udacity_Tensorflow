#ifndef TENSORPOCESSOR_H
#define TENSORPOCESSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <deque>
#include <string>
#include <map>

//#include <memory>
//#include <thread>
#include <condition_variable>
#include <tensorflow/c/c_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

class TensorProcessorClass;

struct DetectionClass
{
  float score;
  int detclass;
  Point2d BoxTopLeft;
  Point2d BoxBottomRigth;
};

class DetectionResultClass
{
private:
  unique_ptr<Mat> _image;
  int _num_detections;
  mutex _mtx;

protected:
  vector<DetectionClass> _detections;

  friend TensorProcessorClass;

public:
  const Mat &GetImage() { return *_image; }
  const vector<DetectionClass> &GetDetections() const { return _detections; }

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
  TensorProcessorClass(const string saved_model_dir,
                       const int NumInputs,
                       const string Input_Tensor_Name,
                       const int NumOutputs,
                       const string Output_Tensor_Name);
  ~TensorProcessorClass();

  void SessionRunLoop();
  const string &GetStringFromClass(int classID) const { return _detClasses.find(classID)->second; }

  const map<int, string> &GetDetClasses() const { return _detClasses; }

  MessageQueue<DetectionResultClass> input_queue;
  MessageQueue<DetectionResultClass> output_queue;

private:
  /*
  unique_ptr<TF_Graph> _graph;

  unique_ptr<TF_SessionOptions> _sessionOpts;
  unique_ptr<TF_Buffer> _runOpts;
  unique_ptr<TF_Session> _session;
  unique_ptr<TF_Status> _status;
  */

  TF_Graph *_graph;
  TF_SessionOptions *_sessionOpts;
  TF_Buffer *_runOpts;
  TF_Session *_session;
  TF_Status *_status;
  TF_Output *_input;
  TF_Output *_output;
  map<int, string> _detClasses;

  const int _numInputs;
  const int _numOutputs;
};

#endif