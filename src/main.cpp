#include <stdio.h>
#include <stdlib.h>

#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <iomanip>
#include <iterator>
//#include <future>

#include <mqtt/client.h>
#include <sstream>

#include "VideoReader.h"
#include "VideoServer.h"
#include "RTSPServer.h"
#include "TensorLiteProcessor.h"
#include "MessageQueue.h"
#include "MessageQueue.cpp" //neded to avoid linker issues
#include "ResultHandling.h"

using namespace cv;
using namespace std;

bool haveDisplay = false;

MessageQueue<unique_ptr<Mat>> RTSPServerClass::input_queue = MessageQueue<unique_ptr<Mat>>();
unique_ptr<Mat> RTSPServerClass::_current_frame = nullptr;
int RTSPServerClass::avgframecycle = 0;
int RTSPServerClass::rtsp_frames = 0;

int main(int argc, char *argv[])
{

  setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp", 1);

  string InstanceName = "DetectorPi_Guard";
  //string PathToModel = "../ssd_mobilenet_v2";
  string PathToModel = "../lite-model_ssd_mobilenet_v1_1_metadata_2.tflite";
  string PathToLabelmap = "../labelmap.txt";

  int numThreads = 2;
  bool allow16Bitprec = true;

  string Cam_URL = "rtspsrc location=rtsp://192.168.0.49:554/ch0_1.h264 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink drop=true max-buffers=2";

  float scale_factor = 1.0;
  string dest_IP = "192.168.178.36";

  float detection_threshold = 0.5; //minimum confidence to process a detection
  float boxwidth_threshold = 0.5;  //maximum size of boxes to filter out huge boundinb boxes
  float overlap_threshold = 0.7;   //overlap threshold to remove detections boxes
  int waitInMainLoop = 200;

  //string OutputPipe = "appsrc ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast key-int-max=2 ! h264parse ! rtph264pay config-interval=5 pt=96 ! udpsink host=";
  string OutputPipe = "appsrc name=mysrc ! videoconvert ! videorate ! omxh264enc ! video/x-h264,profile=baseline,framerate=30/1 ! rtph264pay name=pay0 pt=96";
  //int OutputPort = 5000;
  bool sendDetectorFrame = true;

  string MQTTuser = "mqtt";
  string MQTTpassword = "double45double";

  cout << "\n Instance " << argv[0] << "is starting.\n";
  if (argc > 1)
  {
    cout << "Start Parameters are:\n";
    copy(argv + 1, argv + argc, std::ostream_iterator<const char *>(std::cout, " "));

    int i = 1;
    InstanceName = argv[i++];
    PathToModel = argv[i++];
    PathToLabelmap = argv[i++];
    numThreads = stoi(argv[i++]);
    allow16Bitprec = (string(argv[i++]) == "true");
    scale_factor = stof(argv[i++]);
    dest_IP = argv[i++];
    detection_threshold = stof(argv[i++]);
    boxwidth_threshold = stof(argv[i++]);
    overlap_threshold = stof(argv[i++]);
    waitInMainLoop = stoi(argv[i++]);
    OutputPipe = argv[i++];
    //OutputPort = stoi(argv[i++]);
    sendDetectorFrame = (string(argv[i++]) == "true");
    MQTTuser = argv[i++];
    MQTTpassword = argv[i++];
    Cam_URL = argv[i++];
  }
  else
    cout << "Not enough arguments provided. Using standard values: " << InstanceName << "\n"
         << PathToModel << "\n"
         << PathToLabelmap << "\n"
         << numThreads << "\n"
         << allow16Bitprec << "\n"
         << scale_factor << "\n"
         << dest_IP << "\n"
         << detection_threshold << "\n"
         << boxwidth_threshold << "\n"
         << overlap_threshold << "\n"
         << waitInMainLoop << "\n"
         << OutputPipe << "\n"
         //<< OutputPort << "\n"
         << sendDetectorFrame << "\n"
         << MQTTuser << "\n"
         << MQTTpassword << "\n"
         << Cam_URL << "\n"
         << "\nParameters are: InstanceName PathToModel PathToLabelmap numThreads allow16Bitprec scale_factor HomeAssistantServerIP detection_threshold boxwidth_threshold  overlap_threshold wait_ms_in_Main_Loop OutputPipe sendDetectorFrame MQTTuser MQTTpassword CameraURL" << endl;

  //cout << cv::getBuildInformation();
  //return 0;

  //detect display
  char *val = getenv("DISPLAY");
  haveDisplay = (val != NULL);

  /*

 VideoCapture cap(RTSP_URL, CAP_GSTREAMER);
  
  //VideoCapture cap(RTSP_URL, CAP_FFMPEG);
  if (!cap.isOpened()) {
      std::cout << "Cannot open RTSP stream" << std::endl;
      return -1;
  }

  Mat rawframe,frame;

  VideoWriter video;

  int frame_count =0;
  while (true) {
      cap >> rawframe;
      if (haveDisplay)
        imshow("RTSP stream", rawframe);
      else
      {
        if (rawframe.empty())
        {
          cout << "No more frames! \n";
          return -1;
        }

          int d_width = rawframe.size[1] * scale_factor;
          int d_height = rawframe.size[0] * scale_factor;

        resize(rawframe, frame, Size(d_width,d_height));

        if (frame_count %100 == 0)
          cout << "Frame #" << frame_count << " read\n";
      }
      if (waitKey(1) == 27)
          break;

      if (!video.isOpened())
      {
        cout << "Try to open Frame Writer\n";
        //x264enc tune=zerolatency speed-preset=ultrafast
        //video.open("appsrc ! videoconvert ! x264enc me=hex tune=zerolatency speed-preset=ultrafast key-int-max=50 intra-refresh=true ! h264parse ! rtph264pay config-interval=5 pt=96 ! udpsink host=" + dest_IP+ " port=5000 sync=false",0, 20, frame.size(), true);
        //video.open("appsrc ! videoconvert ! x264enc me=hex tune=zerolatency bitrate =500 speed-preset=superfast ! h264parse ! rtph264pay config-interval=5 pt=96 ! udpsink host=" + dest_IP+ " port=5000 sync=false",0, 20, frame.size(), true);
        int fourcc =    VideoWriter::fourcc('H','2','6','4');
        string inputPipe= argv[1] + dest_IP+ " port=5000 sync=true";
        cout << "Input pipe: " << inputPipe << endl;

        //video.open("appsrc ! videoconvert ! x264enc me=hex tune=zerolatency bitrate =500 speed-preset=superfast ! h264parse ! rtph264pay config-interval=5 pt=96 ! udpsink host=" + dest_IP+ " port=5000 sync=false",CAP_GSTREAMER,fourcc, 5, frame.size(), true);
      
        video.open(inputPipe ,CAP_GSTREAMER,fourcc, 5, frame.size(), true);
        
        if (!video.isOpened())
        {
          cout << "Frame Writer not availbale!\n";
          return -1;
        }
      }

      video.write(frame);
      frame_count++;
    }
 
    cap.release();
    destroyAllWindows();
 
    return 0;


 VideoWriter video;
*/

  //create Detector Instance of ssd_mobilenet_v2 via separate thread
  //promise<shared_ptr<MobilenetV2Class>> promMobilenet;
  //future<shared_ptr<MobilenetV2Class>> futMobilenet = promMobilenet.get_future();

  //get Detector Objet from thread
  //shared_ptr<MobilenetV2Class> MobilenetV2 = make_shared<MobilenetV2Class>(PathToModel);
  shared_ptr<MobilenetV1Class> MobilenetV1 = make_shared<MobilenetV1Class>(PathToModel, PathToLabelmap);

  //String PathToModel = "../SSDMobilenetOpenImages4";
  //shared_ptr<MobilenetV2_OIv4Class> MobilenetV2 = make_shared<MobilenetV2_OIv4Class>(PathToModel);

  //create instance of video reader and pass URL of RTSP Stream
  VideoReader Reader(Cam_URL, scale_factor);
  Reader.StartGrabberThread();

  RTSPServerClass RTSPServer(OutputPipe);

  //create VideoServer
  /*VideoServerClass VideoServer(dest_IP, OutputPipe, OutputPort);
  if (sendDetectorFrame)
    VideoServer.StartVideoServerThread();
  */
  //Class to manage all results incl sending via MQTT
  ResultHandlerClass ResultHandler(dest_IP, sendDetectorFrame, MQTTuser, MQTTpassword, InstanceName);

  //give thread som tme to instanciate
  //std::this_thread::sleep_for(std::chrono::milliseconds(1));

  //load Model into processor & start detector thread
  //TensorProcessorClass TensorProcessor(MobilenetV2);
  //TensorProcessor.StartProcessorThread();
  TensorLiteProcessorClass TensorProcessor(MobilenetV1, numThreads, allow16Bitprec);
  TensorProcessor.StartProcessorThread();

  //counter for average runtime of main loop
  chrono::milliseconds dur(20);

  //frame counter
  int c1 = 0;
  while (true)
  {
    auto t1 = chrono::high_resolution_clock::now();
    unique_ptr<Mat> frame(Reader.getNextFrame());

    //inital feed of detector pipeline
    if (c1 == 0)
    {
      RTSPServer.StartRTSPServerThread(frame->size);
      TensorProcessor.input_queue.sendAndClear(move(frame));
    }

    if (TensorProcessor.output_queue.GetSize() > 0 && c1)
    {
      TensorProcessor.input_queue.sendAndClear(move(frame));
      DetectionResultClass SessionOutput(TensorProcessor.output_queue.receive());
      cout << "Detection Score of ID 0: " << SessionOutput.GetDetections()[0].score << " for " << SessionOutput.GetDetections()[0].ClassName << " at:(" << SessionOutput.GetDetections()[0].BoxTopLeft.x << "," << SessionOutput.GetDetections()[0].BoxTopLeft.y << "),(" << SessionOutput.GetDetections()[0].BoxBottomRigth.x << "," << SessionOutput.GetDetections()[0].BoxBottomRigth.y << ")\n";

      ResultHandler.ResultHandling(SessionOutput, detection_threshold, boxwidth_threshold, overlap_threshold);
      if (sendDetectorFrame)
        RTSPServerClass::input_queue.sendAndClear(move(SessionOutput.MoveImage()));
    }

    auto t2 = chrono::high_resolution_clock::now();
    auto ms_int = chrono::duration_cast<chrono::milliseconds>(t2 - t1);
    dur = (dur + ms_int) / 2;

    if (c1 % 100 == 0)
    {
      cout << "\n---------------------------- \nAvergage runtime of Main loop: " << dur.count() << "ms\n";
      //cout << "\n---------------------------- \n Avergage call frequency of Main loop: " << dur.count() << "ms\n";
      cout << "Avergage call frequency after " << RTSPServerClass::rtsp_frames << " frames of Need Data: " << RTSPServerClass::avgframecycle << "ms\n";
    }
    c1++;
    waitKey(waitInMainLoop);
  }

  /*
  //give thread som tme to instanciate
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  //create mqtt client object
  mqtt::client cli{"tcp://" + dest_IP + ":1883", "DetectorPi"};

  while (true)
  {
    unique_ptr<Mat> frame(move(Reader.getNextFrame()));

    // Check if Frame is empty
    if (frame->empty())
    {
      cout << "No more Frames\n";
      if (haveDisplay)
        waitKey();

      //Stop Threads
      cout << "Stopping Threads\n";
      Reader.StopGrabberThread();
      VideoServer.StopVideoServerThread();
      TensorProcessor.StopProcessorThread();

      // When everything done, release the video window
      destroyAllWindows();
      return 0;
    }

    //check if we have new detection results or if we are in the first run
    if (TensorProcessor.output_queue.GetSize() > 0 || c1 == 0)
    {

      cout << "Frame Number " << c1 << " is send to Detector \n";
      //move frameinto que to be processed by tensor flow
      TensorProcessor.input_queue.sendAndClear(move(frame));

      if (c1 != 0) //skip first itteration
      {
        //get Image from detection and display
        DetectionResultClass SessionOutput(move(TensorProcessor.output_queue.receive()));
        cout << "\nDone\n";
        cout << "Detection Score of ID 0: " << SessionOutput.GetDetections()[0].score << " for " << SessionOutput.GetDetections()[0].ClassName << " at TopLeft Postion: " << SessionOutput.GetDetections()[0].BoxTopLeft.x << "," << SessionOutput.GetDetections()[0].BoxTopLeft.y << "\n";
      }
    }
    else
    {

      //  imshow("Frame", *frame);
      if (c1 % 10 == 0)
      {
        cout << "Frame Number " << c1 << " with size: " << frame->size[0] << " is send to VideoServer \n";
        VideoServer.input_queue.sendAndClear(move(frame));
      }
    }

    waitKey(waitInMainLoop);
    c1++;
  }*/
}
