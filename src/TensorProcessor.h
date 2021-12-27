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
#include "MessageQueue.h"
#include "MessageQueue.cpp"

using namespace std;
using namespace cv;

class TensorProcessorClass;
class DetectorClass;
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

  //friend TensorProcessorClass; //make it friend to allow acess from Processor Class to Detections (but not other classes)

public:
  const Mat &GetImage() { return *_image; }
  const vector<Detection_t> &GetDetections() const { return _detections; }

  DetectionResultClass(unique_ptr<Mat> Image) : _image(move(Image)){};
  void AddDetection(Detection_t det) { _detections.emplace_back(det); }

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

class TensorDescription
{

public:
  //TensorDescription() : Tag(""), Type(TF_FLOAT), ID(0), dims(0),channel(0){};
  TensorDescription() = delete;
  /*TensorDescription(const string tag, TF_DataType type, const int id, const vector<int64_t> Dims) : Tag(tag),
                                                                                                    Type(type),
                                                                                                    ID(id),
                                                                                                    dims(dims),
                                                                                                    channel(0),
                                                                                                    Width(-1),
                                                                                                    Height(-1){};*/
  TensorDescription(const string tag, TF_DataType type, const int id, const vector<int64_t> Dims, int chan = 0) : Tag(tag),
                                                                                                                  Type(type),
                                                                                                                  ID(id),
                                                                                                                  dims(Dims),
                                                                                                                  Width(-1),
                                                                                                                  Height(-1),
                                                                                                                  channel(chan){};

  ~TensorDescription(){};
  const int ID;      //ID in vector to find it ofter detection
  const int channel; //challe of output vector (used if output name is slways the same)
  vector<int64_t> dims;

  const string Tag;
  const TF_DataType Type;
  int Width;
  int Height;

  /*
  //move assign
  TensorDescription &operator=(TensorDescription &&source):Tag(source.Tag)
  {
    if (this == &source)
      return *this;

    Tag = source.Tag;
    Type = source.Type;
    ID = source.ID;

    source.ID = -1;
    source.Tag = "";
    return *this;
  }
  //copy assignment operator
  TensorDescription &operator=(TensorDescription &source)
  {
    if (this == &source)
      return *this;

    Tag = source.Tag;
    Type = source.Type;
    ID = source.ID;

    source.ID = -1;
    source.Tag = "";
    return *this;
  }

  //move assign
  TensorDescription &operator=(TensorDescription &&source)
  {
    if (this == &source)
      return *this;

    Tag = source.Tag;
    Type = source.Type;
    ID = source.ID;

    source.ID = -1;
    source.Tag = "";
    return *this;
  }
  //copy assignment operator
  TensorDescription &operator=(TensorDescription &source)
  {
    if (this == &source)
      return *this;

    Tag = source.Tag;
    Type = source.Type;
    ID = source.ID;

    source.ID = -1;
    source.Tag = "";
    return *this;
  }
  */
};

// generic Detector Class
class DetectorClass
{
protected:
  DetectorClass(string PtoModel, TensorDescription inputDesc) : _pathToModel(PtoModel),
                                                                _inputTensorDescription(inputDesc){};

public:
  DetectorClass() = delete;
  ~DetectorClass(){};

  virtual const string GetDetectorName() = 0;
  virtual unique_ptr<char> ConvertImage(const Mat &OpenCVImage) = 0;

  //const int _numInputs =1;
  //const string _nameOfOutputs;
  //const string _nameOfInputs;
  const string _pathToModel;

  //void AddOutputTensorDescription(TensorDescription desc) { _outputTensorDescriptions.emplace_back(desc); }
  //void SetInputTensorDescription(TensorDescription desc) { _inputTensorDescription = desc; }

  virtual DetectionResultClass ProcessResults(DetectionResultClass &&SessionResult, TF_Tensor **OutputValues, int width, int heigth) = 0;

  virtual void SetImageSize(const int w, const int h) = 0;
  virtual const int GetImageWidth() = 0;
  virtual const int GetImageHeigth() = 0;

  vector<TensorDescription> &GetOutputTensorDescriptions() { return _outputTensorDescriptions; }
  const TensorDescription &GetInputTensorDescription() { return _inputTensorDescription; }

  const int GetInputTensorSize() { return 1; }

protected:
  vector<TensorDescription> _outputTensorDescriptions;
  TensorDescription _inputTensorDescription;
};

//SSD+MobileNetV2 network trained on Open Images V4.
// https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1
class MobilenetV2_OIv4Class : public DetectorClass
{
public:
  const string tag_set = "";
  const string signature_def = "default";

  const string GetDetectorName() override
  {
    return "SSD+MobileNetV2 network trained on Open Images V4";
  }

  MobilenetV2_OIv4Class(string &PathToModel) : DetectorClass(PathToModel, TensorDescription("input_tensor", TF_FLOAT, 0, {-1, -1, 3}))
  {
    //input Tensor is already created at construction - outputs later
    //create output Tensors and put their descrition into a vector for tensor processor to init
    int i = 0;
    TensorDescription detection_boxes("strided_slice", TF_FLOAT, i++, {4});
    _outputTensorDescriptions.push_back(detection_boxes);

    TensorDescription detection_class_entities("strided_slice", TF_FLOAT, i++, {1});
    _outputTensorDescriptions.push_back(detection_class_entities);

    TensorDescription detection_class_labels("strided_slice_2", TF_STRING, i++, {1});
    _outputTensorDescriptions.push_back(detection_class_labels);

    TensorDescription detection_class_names("index_to_string_1_Lookup", TF_STRING, i++, {1});
    _outputTensorDescriptions.push_back(detection_class_names);

    TensorDescription detection_scores("strided_slice_1", TF_FLOAT, i++, {1});
    _outputTensorDescriptions.push_back(detection_scores);
  };
  ~MobilenetV2_OIv4Class(){};

  unique_ptr<char> ConvertImage(const Mat &OpenCVImage) override
  {
    Mat img2;
    OpenCVImage.convertTo(img2, CV_32FC3, 1.f / 255);
    return (make_unique<char>(*img2.data));
  }

  DetectionResultClass ProcessResults(DetectionResultClass &&SessionResult, TF_Tensor **OutputValues, int width, int heigth) override
  {

    return (move(SessionResult));
  }
};

//specific class for Mobilenet containing its config
// https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2
class MobilenetV2Class : public DetectorClass
{
public:
  MobilenetV2Class(string &PathToModel) : DetectorClass(PathToModel, TensorDescription("serving_default_input_tensor", TF_UINT8, 0, {1, -1, -1, 3}))
  {

    //create output Tensors and put their descrition into a vector for tensor processor to init
    int i = 0;

    TensorDescription num_detections("StatefulPartitionedCall", TF_INT32, i++, {1}, 0);
    _outputTensorDescriptions.push_back(num_detections);

    TensorDescription detection_boxes("StatefulPartitionedCall", TF_FLOAT, i++, {4}, 1);
    _outputTensorDescriptions.push_back(detection_boxes);

    TensorDescription detection_classes("StatefulPartitionedCall", TF_INT32, i++, {1}, 2);
    _outputTensorDescriptions.push_back(detection_classes);

    TensorDescription detection_scores("StatefulPartitionedCall", TF_FLOAT, i++, {1}, 3);
    _outputTensorDescriptions.push_back(detection_scores);

    TensorDescription raw_detection_boxes("StatefulPartitionedCall", TF_FLOAT, i++, {3}, 4);
    _outputTensorDescriptions.push_back(raw_detection_boxes);

    TensorDescription raw_detection_scores("StatefulPartitionedCall", TF_FLOAT, i++, {3}, 5);
    _outputTensorDescriptions.push_back(raw_detection_scores);

    TensorDescription detection_anchor_indices("StatefulPartitionedCall", TF_FLOAT, i++, {1}, 6);
    _outputTensorDescriptions.push_back(detection_anchor_indices);

    TensorDescription detection_multiclass_scores("StatefulPartitionedCall", TF_FLOAT, i++, {3}, 7);
    _outputTensorDescriptions.push_back(detection_multiclass_scores);

    string filename = _pathToModel + "/" + "mscoco_label_map.pbtxt";

    _detClasses = ReadClasses2Labels(filename);
  }
  ~MobilenetV2Class(){};

  void SetImageSize(const int w, const int h)
  {
    _inputTensorDescription.Width = w;
    _inputTensorDescription.Height = h;
    _inputTensorDescription.dims = {1, h, w, 3};
  }

  const int GetImageWidth() { return _inputTensorDescription.Width; }
  const int GetImageHeigth() { return _inputTensorDescription.Height; }

  const string GetDetectorName() override
  {
    return "MobilenetV2Class";
  }

  unique_ptr<char> ConvertImage(const Mat &OpenCVImage) override
  {
    return (make_unique<char>(*OpenCVImage.data));
  }

  DetectionResultClass ProcessResults(DetectionResultClass &&SessionResult, TF_Tensor **OutputValues, int width, int heigth) override
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

    return (move(SessionResult));
  }

private:
  map<int, string> _detClasses;
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