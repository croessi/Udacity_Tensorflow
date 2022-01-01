#include "TensorProcessor.h"

#include <iostream>

void NoOpDeallocator(void *data, size_t a, void *b) {}

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
  const char *tags = _detector->tag.c_str(); // default model serving tag; can change in future

  //char i = 0;
  //while (i < 256)
  //{

    //const char tags1[] = {i, 0x00};
    //const char *tags = &tags1[0];
    //const char *tags = """";
    int ntags = 1;
    //i++;
    _session = TF_LoadSessionFromSavedModel(_sessionOpts, _runOpts, _detector->_pathToModel.c_str(), &tags, ntags, _graph, NULL, _status);
    if (TF_GetCode(_status) == TF_OK)
    {
      cout << "Loading Model " << _detector->GetDetectorName() << " from: " << _detector->_pathToModel << " was successfull\n";
      //break;
    }
    else
      cout << "Loading Model " << _detector->GetDetectorName() << " from: " << _detector->_pathToModel << " FAILED!!!!!" << TF_Message(_status);
  //}

  //****** Get input tensor
  _input = (TF_Output *)malloc(sizeof(TF_Output) * _detector->GetInputTensorSize());

  TF_Output t0 = {TF_GraphOperationByName(_graph, _detector->GetInputTensorDescription().Tag.c_str()), 0};
  if (t0.oper == NULL)
  {
    cout << "ERROR: Failed finding Input Tensor\n";
  }
  else
    cout << "Input Tensor created and assigned\n";

  _input[0] = t0;

  //********* Get Output tensor
  _output = (TF_Output *)malloc(sizeof(TF_Output) * _detector->GetOutputTensorDescriptions().size());

  for (TensorDescription &outDesc : _detector->GetOutputTensorDescriptions())
  {
    TF_Output t2 = {TF_GraphOperationByName(_graph, outDesc.Tag.c_str()), outDesc.ID};
    if (t2.oper == NULL)
      cout << "ERROR: Failed finding Output Tensor with Tag : " << outDesc.Tag << endl;

    _output[outDesc.ID] = t2;
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
  TF_Tensor **InputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * _detector->GetInputTensorSize());
  TF_Tensor **OutputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * _detector->GetOutputTensorDescriptions().size());

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

    int height = SessionResult.GetImage().size[0];
    int width = SessionResult.GetImage().size[1];

    //int64_t dims[] = {1, SessionResult.GetImage().size[0], SessionResult.GetImage().size[1], 3};
    //int ndata = dims[0] * dims[1] * dims[2] * dims[3];

    if (_detector->GetImageHeigth() == -1 || _detector->GetImageWidth() == -1)
      _detector->SetImageSize(width, height);

    TensorDescription InputDesc = _detector->GetInputTensorDescription();

    int ndata = InputDesc.dims.size() * height * width * TF_DataTypeSize(InputDesc.Type);

    //transform image to make it fit for the input tensor
    unique_ptr<Mat> InputImage = _detector->ConvertImage(SessionResult.GetImage());

    TF_Tensor *int_tensor = TF_NewTensor(_detector->GetInputTensorDescription().Type, InputDesc.dims.data(), InputDesc.dims.size(), InputImage->data, ndata, &NoOpDeallocator, 0);

    if (int_tensor == NULL)
      printf("ERROR: Failed to contruct Input Tensor\n");

    InputValues[0] = int_tensor;

    //Run the Session
    auto t1 = chrono::high_resolution_clock::now();

    TF_SessionRun(_session, NULL, _input, InputValues, _detector->GetInputTensorSize(), _output, OutputValues, _detector->GetOutputTensorDescriptions().size(), NULL, 0, NULL, _status);

    InputImage->release();
    
    auto t2 = chrono::high_resolution_clock::now();
    auto ms_int = chrono::duration_cast<chrono::milliseconds>(t2 - t1);

    if (TF_GetCode(_status) == TF_OK)
      cout << "Session ran through in " << ms_int.count() << "ms\n";
    else
      cout << "Session run error :" << TF_Message(_status);

    SessionResult = _detector->ProcessResults(move(SessionResult), OutputValues, width, height);

    //move back detection via receive que
    output_queue.send(move(SessionResult));
  }
}
