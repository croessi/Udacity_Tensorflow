#include <stdio.h>
#include <stdlib.h>

#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <future>

#include "VideoReader.h"
#include "TensorProcessor.h"
#include "MessageQueue.h"
#include "MessageQueue.cpp" //neded to avoid linker issues

using namespace cv;
using namespace std;

const float display_threshold = 0.5;  //minimum confidence to display a bounding box
const float boxwidth_threshold = 0.5; //maximum size of boxes to filter out huge boundinb boxes

bool haveDisplay = false;

int main()
{

  //detect display
  char *val = getenv("DISPLAY");
  haveDisplay = (val != NULL);

  //create Detector Instance of ssd_mobilenet_v2 via separate thread
  //promise<shared_ptr<MobilenetV2Class>> promMobilenet;
  //future<shared_ptr<MobilenetV2Class>> futMobilenet = promMobilenet.get_future();

  String PathToModel = "../ssd_mobilenet_v2";
  string PathToLabels = PathToModel + "/" + "mscoco_label_map.pbtxt";

   //get Detector Objet from thread
  shared_ptr<MobilenetV2Class> MobilenetV2 = make_shared<MobilenetV2Class>(PathToModel);
 
  //thread MobilenetThread(ReturnMobilenetV2, move(promMobilenet),PathToModel,PathToLabels);


  //create instance of video reader
  VideoReader Reader("../output.mp4");

  Reader.StartGrabberThread();

  //give thread som tme to instanciate
  std::this_thread::sleep_for(std::chrono::milliseconds(1));



  //load Model into processor
  TensorProcessorClass TensorProcessor(MobilenetV2);

  //start detector thread
  TensorProcessor.StartProcessorThread();

  //give thread som tme to instanciate
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  //frame counter
  int c1 = 0;
  while (true)
  {

    unique_ptr<Mat> frame(move(Reader.getNextFrame()));

    // Display the resulting frame - if a display and frame are available
    if (frame->empty())
    {
      cout << "No more Frames\n";
      if (haveDisplay)
        waitKey();

      //Stop Threads
      cout << "Stopping Threads\n";
      Reader.StopGrabberThread();
      TensorProcessor.StopProcessorThread();

      // When everything done, release the video window
      destroyAllWindows();
      return 0;
    }

    //check if we have new images or if we are in the first run
    if (TensorProcessor.output_queue.GetSize() > 0 || c1 <= 1)
    {
      //move frame into detection result
      DetectionResultClass SessionInput(move(frame));
      cout << "Frame Number " << c1 << " is send to Detector \n";
      //move detection result into que to be processed by tensor flow
      TensorProcessor.input_queue.send(move(SessionInput));

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
        cout << "Detection Score of ID 0: " << SessionOutput.GetDetections()[0].score << " for " << SessionOutput.GetDetections()[0].ClassName << " at TopLeft Postion: " << SessionOutput.GetDetections()[0].BoxTopLeft.x << "," << SessionOutput.GetDetections()[0].BoxTopLeft.y << "\n";

        char buffer[100];
        for (Detection_t d : SessionOutput.GetDetections())
        {
          float boxwidth = (d.BoxBottomRigth.x - d.BoxTopLeft.x) / (float)SessionOutput.GetImage().size[1];

          if (d.score > display_threshold && boxwidth < boxwidth_threshold)
          {
            rectangle(SessionOutput.GetImage(), d.BoxTopLeft, d.BoxBottomRigth, Scalar(0, 255, 0), 1, 8, 0);
            

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

        // Display the resulting frame - if a display is available
        if (haveDisplay)
        {
          imshow("DetectorResults", SessionOutput.GetImage());
          //waitKey(0);
        }
      }
    }
    else
    {
      //if (haveDisplay)
      //  imshow("Frame", *frame);
      cout << "Frame Number " << c1 << " is displayed \n";
      waitKey(30);
    }
    c1++;
  }
  return 0;
}
