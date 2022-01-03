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

class TensorDescription
{

public:
  TensorDescription() = delete;
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
};

// generic Detector Class
class DetectorClass
{
protected:
  DetectorClass(string PtoModel, const string Tag, const string Signature, TensorDescription inputDesc) : _pathToModel(PtoModel),
                                                                                                          _inputTensorDescription(inputDesc),
                                                                                                          tag(Tag),
                                                                                                          signature(Signature){};

public:
  DetectorClass() = delete;
  ~DetectorClass(){};

  virtual const string GetDetectorName() = 0;
  virtual void SetImageSize(const int w, const int h) = 0;
  virtual unique_ptr<Mat> ConvertImage(const Mat &OpenCVImage) = 0;
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

  MobilenetV2_OIv4Class(string &PathToModel) : DetectorClass(PathToModel, " ", "default", TensorDescription("input_tensor", TF_FLOAT, 0, {-1, -1, 3}))
  {
    //parameters can bea read out via gogle colar with !saved_model_cli show --dir {'Model'}  --tag_set serve --signature_def serving_default
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

  void SetImageSize(const int w, const int h)
  {
    _inputTensorDescription.Width = w;
    _inputTensorDescription.Height = h;
    _inputTensorDescription.dims = {h, w, 3};
  }

  unique_ptr<Mat> ConvertImage(const Mat &OpenCVImage) override
  {
    Mat img2;
    OpenCVImage.convertTo(img2, CV_32FC3, 1.f / 255);
    return (make_unique<Mat>(img2));
  }

  void ProcessResults(DetectionResultClass &SessionResult, TF_Tensor **OutputValues, int width, int heigth) override
  {
  }
};

//specific class for Mobilenet containing its config
// https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2
class MobilenetV2Class : public DetectorClass
{
public:
  MobilenetV2Class(string &PathToModel) : DetectorClass(PathToModel, "serve", "serving_default", TensorDescription("serving_default_input_tensor", TF_UINT8, 0, {1, -1, -1, 3}))
  {

    //parameters can bea read out via gogle colar with !saved_model_cli show --dir {'Model'}  --tag_set serve --signature_def serving_default
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

class TensorProcessorClass
{
public:
  TensorProcessorClass(shared_ptr<DetectorClass> Detector);
  ~TensorProcessorClass();

  void StartProcessorThread();
  void StopProcessorThread();

  MessageQueue<unique_ptr<Mat>> input_queue;
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