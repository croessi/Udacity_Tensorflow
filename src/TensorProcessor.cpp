#include "TensorProcessor.h"

#include <iostream>

void NoOpDeallocator(void *data, size_t a, void *b) {}

/*_graph(make_unique<TF_Graph>(TF_NewGraph())),
                                                                              _sessionOpts(make_unique<TF_SessionOptions>(TF_NewSessionOptions())),
                                                                              _runOpts(nullptr),
                                                                              _numInputs(NumInputs),
                                                                              _numOutputs(NumOutputs)
                                                                              */

void TensorProcessorClass::StartProcessorThread()
{
  _detectorThread = thread(&TensorProcessorClass::SessionRunLoop, this);
}
void TensorProcessorClass::StopProcessorThread()
{
  //create empty frame and set it to queue to trigger thread exit
  unique_ptr<Mat> frame = make_unique<Mat>(Mat(Size(0, 0), CV_8U));
  DetectionResultClass det(move(frame));
  input_queue.send(move(det));
  cout << "Waiting for Detector thread to join.\n";
  _detectorThread.join();
}

TensorProcessorClass::TensorProcessorClass(shared_ptr<DetectorClass> Detector) : _detector(Detector)
{

  _graph = TF_NewGraph();
  _status = TF_NewStatus();
  _sessionOpts = TF_NewSessionOptions();
  _runOpts = nullptr;

  //********* Read model
  const char *tags = "serve"; // default model serving tag; can change in future
  int ntags = 1;

  _session = TF_LoadSessionFromSavedModel(_sessionOpts, _runOpts, _detector->_pathToModel.c_str(), &tags, ntags, _graph, NULL, _status);
  if (TF_GetCode(_status) == TF_OK)
    cout << "Loading Model " << _detector->GetDetectorName() << " from: " << _detector->_pathToModel << " was successfull\n";
  else
    cout << "Loading Model " << _detector->GetDetectorName() << " from: " << _detector->_pathToModel << " FAILED!!!!!" << TF_Message(_status);

  //****** Get input tensor
  _input = (TF_Output *)malloc(sizeof(TF_Output) * _detector->_numInputs);

  TF_Output t0 = {TF_GraphOperationByName(_graph, _detector->_nameOfInputs.c_str()), 0};
  if (t0.oper == NULL)
  {
    cout << "ERROR: Failed finding Input Tensor\n";
  }
  else
    cout << "Input Tensor created and assigned\n";

  _input[0] = t0;

  //********* Get Output tensor
  _output = (TF_Output *)malloc(sizeof(TF_Output) * _detector->_numOutputs);
  //vector<TF_Output> _output(_numOutputs);

  for (int i = 0; i < _detector->_numOutputs; i++)
  {
    TF_Output t2 = {TF_GraphOperationByName(_graph, _detector->_nameOfOutputs.c_str()), i};
    if (t2.oper == NULL)
      cout << "ERROR: Failed finding Output Tensor Nr: " << i << endl;

    _output[i] = t2;
  }

  printf("Output Tensors created and assigned\n");
}

TensorProcessorClass::~TensorProcessorClass()
{
  free(_input);
  free(_output);
  TF_DeleteGraph(_graph);
  TF_DeleteSession(_session, _status);
  TF_DeleteSessionOptions(_sessionOpts);
  TF_DeleteStatus(_status);
}

void TensorProcessorClass::SessionRunLoop()
{
  //********* Allocate data for inputs & outputs
  TF_Tensor **InputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * _detector->_numInputs);
  TF_Tensor **OutputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * _detector->_numOutputs);

  while (true)
  {
    //wait for empty detection to be received
    DetectionResultClass SessionResult = input_queue.receive();

    //empty frame --> cleanup and exit thread
    if (SessionResult.GetImage().size[0] == 0 || SessionResult.GetImage().size[1] == 0)
    {
      free(InputValues);
      free(OutputValues);
      return;
    } 

    int ndims = 4;

    int64_t dims[] = {1, SessionResult.GetImage().size[0], SessionResult.GetImage().size[1], 3};
    int ndata = dims[0] * dims[1] * dims[2] * dims[3];

    //construct tensor
    //unique_ptr<TF_Tensor> int_tensor = make_unique<TF_Tensor>(TF_NewTensor(TF_UINT8, dims, ndims, detection.GetImage().data, ndata, &NoOpDeallocator, 0));

    TF_Tensor *int_tensor = TF_NewTensor(TF_UINT8, dims, ndims, SessionResult.GetImage().data, ndata, &NoOpDeallocator, 0);

    if (int_tensor == NULL)
      printf("ERROR: Failed to contruct Input Tensor\n");

    InputValues[0] = int_tensor;

    //Run the Session
    TF_SessionRun(_session, NULL, _input, InputValues, _detector->_numInputs, _output, OutputValues, _detector->_numOutputs, NULL, 0, NULL, _status);

    if (TF_GetCode(_status) == TF_OK)
      cout << "Session ran through\n";
    else
      cout << "Session run error :" << TF_Message(_status);

    //decode detection

    int num_detections = *(float *)(TF_TensorData(OutputValues[5]));
    for (int i = 0; i < num_detections; i++)
    {
      Detection_t Detection;
      Detection.score = ((float *)TF_TensorData(OutputValues[4]))[i];
      Detection.detclass = ((float *)TF_TensorData(OutputValues[2]))[i];

      BoundingBox_t box = ((BoundingBox_t *)TF_TensorData(OutputValues[1]))[i];

      Detection.BoxTopLeft.x = (int)(box.x1 * dims[2]);
      Detection.BoxTopLeft.y = (int)(box.y1 * dims[1]);

      Detection.BoxBottomRigth.x = (int)(box.x2 * dims[2]);
      Detection.BoxBottomRigth.y = (int)(box.y2 * dims[1]);

      SessionResult._detections.emplace_back(move(Detection));
    }

    //move back detection via receive que
    output_queue.send(move(SessionResult));

  }
}
