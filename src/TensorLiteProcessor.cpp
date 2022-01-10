#include "TensorLiteProcessor.h"

#include <iostream>

void TensorLiteProcessorClass::StartProcessorThread()
{
  _detectorThread = thread(&TensorLiteProcessorClass::SessionRunLoop, this);
}
void TensorLiteProcessorClass::StopProcessorThread()
{
  //create empty frame and set it to queue to trigger thread exit
  unique_ptr<Mat> frame = make_unique<Mat>(Mat(Size(0, 0), CV_8U));
  input_queue.sendAndClear(move(frame));
  cout << "Waiting for Detector thread to join.\n";
  _detectorThread.join();
}

TensorLiteProcessorClass::TensorLiteProcessorClass(shared_ptr<DetectorClass> Detector) : _detector(Detector)
{

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(_detector->_pathToModel);
  if (model)
    cout << "Loading Model " << _detector->GetDetectorName() << " from: " << _detector->_pathToModel << " was successfull\n";

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);

  builder(&_interpreter);
  if (interpreter)
    cout << "Interpreter for " << _detector->GetDetectorName() << " built successfully!\n";
}

TensorLiteProcessorClass::~TensorLiteProcessorClass()
{
}

void TensorLiteProcessorClass::SessionRunLoop()
{
  //********* Allocate data for inputs & outputs
  TF_Tensor **InputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * _detector->GetInputTensorSize());
  TF_Tensor **OutputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * _detector->GetOutputTensorDescriptions().size());

  //DEBUG

  while (true)
  {
    //wait for frame
    DetectionResultClass SessionResult(move(input_queue.receive()));

    //empty frame --> cleanup and exit thread
    if (SessionResult.GetImage().size[0] == 0 || SessionResult.GetImage().size[1] == 0)
    {
      free(InputValues);
      free(OutputValues);
      return;
    }

    //supporting values
    int height = SessionResult.GetImage().size[0];
    int width = SessionResult.GetImage().size[1];

    //pass image dims to detector
    if (_detector->GetImageHeigth() == -1 || _detector->GetImageWidth() == -1)
      _detector->SetImageSize(width, height);

    _detector->
        //get Input Tensor Description
        TensorDescription InputDesc = _detector->GetInputTensorDescription();
    int ndata = InputDesc.dims.size() * height * width * TF_DataTypeSize(InputDesc.Type);

    //transform image to make it fit for the input tensor
    unique_ptr<Mat> InputImage(_detector->ConvertImage(SessionResult.GetImage()));

    //TF_Tensor *int_tensor = TF_NewTensor(_detector->GetInputTensorDescription().Type, InputDesc.dims.data(), InputDesc.dims.size(), InputImage->data, ndata, &NoOpDeallocator, 0);

    TF_Tensor *int_tensor = TF_NewTensor(_detector->GetInputTensorDescription().Type, InputDesc.dims.data(), InputDesc.dims.size(), SessionResult.GetImage().data, ndata, &NoOpDeallocator, 0);

    if (int_tensor == NULL)
      printf("ERROR: Failed to contruct Input Tensor\n");

    InputValues[0] = int_tensor;

    //Run the Session
    auto t1 = chrono::high_resolution_clock::now();

    // Run inference
    auto InterpreterResut = interpreter->Invoke();

    printf("\n\n=== Post-invoke Interpreter State ===\n");
    tflite::PrintInterpreterState(interpreter.get());

    auto t2 = chrono::high_resolution_clock::now();
    auto ms_int = chrono::duration_cast<chrono::milliseconds>(t2 - t1);

    SessionResult.runtime = (int)ms_int.count();

    if (InterpreterResut == kTfLiteOk))
      cout << "Session ran through in " << ms_int.count() << "ms\n";
    else
      cout << "Session run error :" << TF_Message(_status);

    //interpretation of results
    _detector->ProcessResults(SessionResult, OutputValues, width, height);

    //cleanup
    TF_DeleteTensor(int_tensor);
    for (int i = 0; i < _detector->GetOutputTensorDescriptions().size(); i++)
    {
      TF_DeleteTensor(OutputValues[i]);
    }

    //move back detection via output que
    output_queue.sendAndClear(move(SessionResult));
  }
}
