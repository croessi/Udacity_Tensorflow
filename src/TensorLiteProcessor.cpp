#include "TensorLiteProcessor.h"

#include <iostream>

std::string getShape(TfLiteTensor *t)
{
  std::string s = "(";
  int sz = t->dims->size;
  for (int i = 0; i < sz; i++)
  {
    if (i > 0)
    {
      s += ",";
    }
    s += std::to_string(t->dims->data[i]);
  }
  s += ")";
  return s;
}

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

TensorLiteProcessorClass::TensorLiteProcessorClass(shared_ptr<DetectorLiteClass> Detector, int NumThreads, bool Allow16bitPrecisison) : _detector(Detector),_NumThreads(NumThreads),_Allow16bitPrecisison(Allow16bitPrecisison)
{
}

TensorLiteProcessorClass::~TensorLiteProcessorClass()
{
}

void TensorLiteProcessorClass::SessionRunLoop()
{
  // Load model - has to happen in this thread
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(_detector->_pathToModel.c_str());

  if (model)
    cout << "Loading Model " << _detector->GetDetectorName() << " from: " << _detector->_pathToModel << " was successfull\n";

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);

  builder(&_interpreter);
  if (_interpreter)
    cout << "Interpreter for " << _detector->GetDetectorName() << " built successfully!\n";

  // Get Input and Output tensors info
  int in_id = _interpreter->inputs()[0];
  TfLiteTensor *in_tensor = _interpreter->tensor(in_id);
  auto in_type = in_tensor->type;
  auto in_shape = getShape(in_tensor).c_str();
  auto in_name = in_tensor->name;
  printf("Input Tensor id, name, type, shape: %i, %s, %s(%d), %s\n", in_id, in_name, TfLiteTypeGetName(in_type), in_type, in_shape);

  int out_sz = _interpreter->outputs().size();
  std::cout << "Output Tensor id, name, type, shape:" << std::endl;
  for (int i = 0; i < out_sz; i++)
  {
    auto t_id = _interpreter->outputs()[i];
    TfLiteTensor *t = _interpreter->tensor(t_id);
    auto t_type = t->type;
    printf("  %i, %s, %s(%d), %s\n", t_id, t->name, TfLiteTypeGetName(t_type), t_type, getShape(t).c_str());
  }

  //allocate tesnors in context of sessions run
  if (_interpreter->AllocateTensors() != kTfLiteOk)
  {
    printf("Failed to allocate tensors\n");
    exit(1);
  }
  printf("AllocateTensors Ok\n");

  _interpreter->SetNumThreads(_NumThreads);
  _interpreter->SetAllowFp16PrecisionForFp32(_Allow16bitPrecisison);

  while (true)
  {
    //wait for frame
    DetectionResultClass SessionResult(move(input_queue.receive()));

    //empty frame --> cleanup and exit thread
    if (SessionResult.GetImage().size[0] == 0 || SessionResult.GetImage().size[1] == 0)
    {
      return;
    }

    //cout << "Feed Interpreter" << endl;
    _detector->FeedInterpreter(*_interpreter.get(), SessionResult.GetImage());

    //Run the Session
    auto t1 = chrono::high_resolution_clock::now();

    // Run inference
    //cout << "Run inference" << endl;
    auto InterpreterResut = _interpreter->Invoke();

    //printf("\n\n=== Post-invoke Interpreter State ===\n");
    //tflite::PrintInterpreterState(_interpreter.get());

    auto t2 = chrono::high_resolution_clock::now();
    auto ms_int = chrono::duration_cast<chrono::milliseconds>(t2 - t1);

    SessionResult.runtime = (int)ms_int.count();

    if (InterpreterResut == kTfLiteOk)
      cout << "Session ran through in " << ms_int.count() << "ms\n";
    else
      cout << "Session run error :" << InterpreterResut;

    //interpretation of results
    _detector->ProcessResults(_interpreter, SessionResult);

    //cleanup

    //move back detection via output que
    output_queue.sendAndClear(move(SessionResult));
  }
}
