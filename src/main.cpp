#include <stdio.h>
#include <stdlib.h>

#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <iomanip>
#include <future>

#include <mqtt/client.h>
#include <sstream>

#include "VideoReader.h"
#include "VideoServer.h"
#include "TensorProcessor.h"
#include "MessageQueue.h"
#include "MessageQueue.cpp" //neded to avoid linker issues^
#include "ResultHandling.h"

using namespace cv;
using namespace std;

bool haveDisplay = false;

/*
std::string gst_pipe_in = "rtspsrc location=rtsp://admin:passwd@192.168.1.30:554/cam/realmonitor?channel=1&subtype=0 ! rtph264depay ! h264parse ! v4l2h264dec ! autovideoconvert ! appsink"; cv::VideoCapture capture(gst_pipe_in,cv::CAP_GSTREAMER);

The VideoWriter Gstreamer pipe for writing the processed video to a file:

std::string motion_writer_pipe = "appsrc ! autovideoconvert ! v4l2h264enc ! h264parse ! mp4mux ! filesink location = " + savedMotionVideoFullPath; cv::VideoWriter motion_writer = cv::VideoWriter(motion_writer_pipe,cv::CAP_GSTREAMER,frames_per_second,frame_size,true);
*/
int main(int argc, char *argv[])
{

  setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp", 1);

  //const string RTSP_URL = "rtsp://192.168.0.49:554/ch0_1.h264";
  const string RTSP_URL = "rtspsrc location=rtsp://192.168.0.49:554/ch0_1.h264 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink drop=true max-buffers=2";

  const float scale_factor = 1.0;
  const string dest_IP = "192.168.178.36";

  const float display_threshold = 0.5;  //minimum confidence to process a detection
  const float boxwidth_threshold = 0.5; //maximum size of boxes to filter out huge boundinb boxes

  if (argc < 4 && false)
  {
    cout << "Not enough arguments provided. Usage main.exe source-rtsp-url scale-factor dest-IP\n";
  }

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
  String PathToModel = "../ssd_mobilenet_v2";
  shared_ptr<MobilenetV2Class> MobilenetV2 = make_shared<MobilenetV2Class>(PathToModel);

  //String PathToModel = "../SSDMobilenetOpenImages4";
  //shared_ptr<MobilenetV2_OIv4Class> MobilenetV2 = make_shared<MobilenetV2_OIv4Class>(PathToModel);

  //create instance of video reader and pass URL of RTSP Stream
  VideoReader Reader(RTSP_URL, scale_factor);
  Reader.StartGrabberThread();

  //create VideoServer
  VideoServerClass VideoServer(dest_IP);
  VideoServer.StartVideoServerThread();

  //Class to manage all results incl sending via MQTT
  ResultHandlerClass ResultHandler(dest_IP);

  //give thread som tme to instanciate
  //std::this_thread::sleep_for(std::chrono::milliseconds(1));

  //load Model into processor & start detector thread
  TensorProcessorClass TensorProcessor(MobilenetV2);
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
      TensorProcessor.input_queue.sendAndClear(move(frame));

    if (TensorProcessor.output_queue.GetSize() > 0 && c1)
    {
      TensorProcessor.input_queue.sendAndClear(move(frame));
      DetectionResultClass SessionOutput(TensorProcessor.output_queue.receive());
      cout << "Detection Score of ID 0: " << SessionOutput.GetDetections()[0].score << " for " << SessionOutput.GetDetections()[0].ClassName << " at TopLeft Postion: " << SessionOutput.GetDetections()[0].BoxTopLeft.x << "," << SessionOutput.GetDetections()[0].BoxTopLeft.y << "\n";

      ResultHandler.ResultHandling(SessionOutput, display_threshold, boxwidth_threshold);
      VideoServer.input_queue.sendAndClear(move(SessionOutput.MoveImage()));
    }
    //check if frame has not been moved to detector -> send
    //if (frame.get())
    //  VideoServer.input_queue.sendAndClear(move(frame));

    auto t2 = chrono::high_resolution_clock::now();
    auto ms_int = chrono::duration_cast<chrono::milliseconds>(t2 - t1);
    dur = (dur + ms_int) / 2;

    if (c1 % 20 == 0)
      cout << "Avergage runtime of Main loop: " << dur.count() << "ms\n";
    c1++;
  }

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

    waitKey(30);
    c1++;
  }

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

    //check if we have new images or if we are in the first run
    if (TensorProcessor.output_queue.GetSize() > 0 || c1 <= 1)
    {
      //move frame into detection result
      //DetectionResultClass SessionInput(move(frame));
      cout << "Frame Number " << c1 << " is send to Detector \n";
      //move detection result into que to be processed by tensor flow
      TensorProcessor.input_queue.sendAndClear(move(frame));

      //skip in init run
      if (c1 == 0)
      {
        cout << "Wait until Network has run for the first time.....\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); //give cout some time
      }

      if (c1 != 1) //skip waiting in second run
      {
        //get Image from detection and display
        DetectionResultClass SessionOutput(TensorProcessor.output_queue.receive());
        //cout << "Detection Score of ID 0: " << SessionOutput.GetDetections()[0].score << " for " << SessionOutput.GetDetections()[0].ClassName << " at TopLeft Postion: " << SessionOutput.GetDetections()[0].BoxTopLeft.x << "," << SessionOutput.GetDetections()[0].BoxTopLeft.y << "\n";

        char buffer[100];
        stringstream RawOutput;
        stringstream NiceOutput;

        StatisticsClass Statistics;

        for (Detection_t d : SessionOutput.GetDetections())
        {

          float boxwidth = (d.BoxBottomRigth.x - d.BoxTopLeft.x) / (float)SessionOutput.GetImage().size[1];
          float boxheight = (d.BoxBottomRigth.y - d.BoxTopLeft.y) / (float)SessionOutput.GetImage().size[0];

          float boxcenterX = d.BoxTopLeft.x / (float)SessionOutput.GetImage().size[1] + boxwidth * 0.5;

          float boxcenterY = d.BoxTopLeft.y / (float)SessionOutput.GetImage().size[0] + boxheight * 0.5;

          //{"Timer1":{"Arm": <status>, "Time": <time>}, "Timer2":{"Arm": <status>, "Time": <time>}}

          if (d.score > display_threshold && boxwidth < boxwidth_threshold)
          {
            RawOutput.setf(ios::fixed);
            RawOutput << "Score: " << (int)(d.score * 100) << "% Class: " << d.ClassName << " at: " << setprecision(2) << boxcenterX << ":" << boxcenterY << "\n";

            NiceOutput << "{\"" << d.ClassName << "\"}:{\"Score:\"" << d.score << "\"},";

            rectangle(SessionOutput.GetImage(), d.BoxTopLeft, d.BoxBottomRigth, Scalar(0, 255, 0), 1, 8, 0);

            Statistics.Stat[d.ClassName]++;

            snprintf(buffer, 100, "%s %d%%", d.ClassName.c_str(), (int)(d.score * 100));

            putText(SessionOutput.GetImage(),
                    buffer,
                    Point2d(d.BoxTopLeft.x, d.BoxBottomRigth.y),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    1, cv::LINE_AA, false);
          }
        }

        //check if we are connected to MQTT server
        if (!cli.is_connected())
        {
          cout << "Not connected to Homeassistant MQTT Server -> try to connect...";
          cli.connect();
          if (cli.is_connected())
            cout << "done \n";
        }

        if (cli.is_connected())
        {
          //cout << "RawOutput: \n" << RawOutput.str() << endl;
          mqtt::message_ptr msg;
          //truncate to max 255 signs for MQTT
          /* if (RawOutput.str().size() > 250)
            msg = mqtt::message::create(TOPIC, RawOutput.str().substr(0, 250));
          else
            msg = mqtt::message::create(TOPIC, RawOutput.str());
*/
          cli.publish(msg);
          cout << "Message: " << RawOutput.str() << endl;

          for (auto const &x : Statistics.Stat)
          {
            if (x.second > 0)
              std::cout << x.first << ':' << x.second << std::endl;
          }
        }

        // Display the resulting frame - if a display is available
        if (haveDisplay)
        {
          imshow("DetectorResults", SessionOutput.GetImage());
          //waitKey(0);
        }

        //send frame to Video Server
        //unique_ptr<Mat> Image(move(SessionOutput.MoveImage()));
        //VideoServer.input_queue.send(move(Image));
      }
    }
    else
    {
      //send frame to Video Server
      // VideoServer.input_queue.send(move(*frame));

      //if (haveDisplay)
      //  imshow("Frame", *frame);
      if (c1 % 20 == 0)
        cout << "Frame Number " << c1 << " is displayed \n";
      waitKey(30);
    }
    c1++;
    frame.release();
  }
  return 0;
}
