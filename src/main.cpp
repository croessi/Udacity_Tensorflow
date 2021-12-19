#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

struct ImHeaderStruct
{
  u_int32_t magic_number;
  u_int32_t numImages;
  u_int32_t numRows;
  u_int32_t numColumns;
};

vector<vector<int8_t>> ImageReader(std::string filename)
{
  std::fstream s(filename); //(filename, s.binary | s.trunc | s.in);
  if (!s.is_open())
  {
    std::cout << "failed to open " << filename << '\n';
  }
  else
  {
    // read header as 32 bit integers

    ImHeaderStruct ImHeader;
    s.read((char *)&ImHeader, sizeof(ImHeaderStruct));

    ImHeader.magic_number = __builtin_bswap32(ImHeader.magic_number);
    ImHeader.numColumns = __builtin_bswap32(ImHeader.numColumns);
    ImHeader.numImages = __builtin_bswap32(ImHeader.numImages);
    ImHeader.numRows = __builtin_bswap32(ImHeader.numRows);

    //allocate memory as vector of vectors
    //vector<vector<int8_t>> ret(ImHeader.numImages);
    vector<vector<int8_t>> ret;
    for (int c = 0; c < ImHeader.numImages; c++)
    {
      //alocate memory for image
      vector<int8_t> inIm(ImHeader.numRows * ImHeader.numColumns);
      s.read((char *)inIm.data(), sizeof(inIm));

      ret.emplace_back(move(inIm));
    }
    return ret;
  }
  return vector<vector<int8_t>>();
}

void NoOpDeallocator(void *data, size_t a, void *b) {}

int main()
{
  vector<vector<int8_t>> InImages(ImageReader("../t10k-images"));

  //********* Read model
  TF_Graph *Graph = TF_NewGraph();
  TF_Status *Status = TF_NewStatus();

  TF_SessionOptions *SessionOpts = TF_NewSessionOptions();
  TF_Buffer *RunOpts = NULL;

  const char *saved_model_dir = "../my_model"; // Path of the model
  const char *tags = "serve";                  // default model serving tag; can change in future
  int ntags = 1;

  TF_Session *Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
  if (TF_GetCode(Status) == TF_OK)
  {
    printf("TF_LoadSessionFromSavedModel OK\n");
  }
  else
  {
    printf("%s", TF_Message(Status));
  }
  //****** Get input tensor
  int NumInputs = 1;
  TF_Output *Input = (TF_Output *)malloc(sizeof(TF_Output) * NumInputs);

  TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_flatten_input"), 0};
  if (t0.oper == NULL)
    printf("ERROR: Failed TF_GraphOperationByName serving_default_flatten_input\n");
  else
    printf("TF_GraphOperationByName serving_default_flatten_input is OK\n");

  Input[0] = t0;

  //********* Get Output tensor
  int NumOutputs = 1;
  TF_Output *Output = (TF_Output *)malloc(sizeof(TF_Output) * NumOutputs);

  TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
  if (t2.oper == NULL)
    printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
  else
    printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

  Output[0] = t2;

  //********* Allocate data for inputs & outputs
  TF_Tensor **InputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * NumInputs);
  TF_Tensor **OutputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * NumOutputs);

  int ndims = 2;
  int64_t dims[] = {28, 28};

  float data[28 * 28];

  for (int i = 0; i < 28 * 28; i++)
  {
    data[i] = InImages[0][i] / 255.0;
  }

  int ndata = sizeof(TF_FLOAT) * dims[0] * dims[1]; // This is tricky, it number of bytes not number of element

  TF_Tensor *int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, &data, ndata, &NoOpDeallocator, 0);
  if (int_tensor != NULL)
  {
    printf("TF_NewTensor is OK\n");
  }
  else
    printf("ERROR: Failed TF_NewTensor\n");

  InputValues[0] = int_tensor;
  // //Run the Session
  TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL, Status);

  if (TF_GetCode(Status) == TF_OK)
  {
    printf("Session is OK\n");
  }
  else
  {
    printf("%s", TF_Message(Status));
  }

  // //Free memory
  TF_DeleteGraph(Graph);
  TF_DeleteSession(Session, Status);
  TF_DeleteSessionOptions(SessionOpts);
  TF_DeleteStatus(Status);

  void *buff = TF_TensorData(OutputValues[0]);
  float *offsets = (float *)buff;
  printf("Result Tensor :\n");
  for (int i = 0; i < 10; i++)
  {
    printf("%f\n", offsets[i]);
  }
  return 0;
}
