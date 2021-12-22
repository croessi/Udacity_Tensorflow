# Udacity_Tensorflow

This program asynchronously reads in video frames via Opencv / FFMPEG and sends it to a second thread with a tensorflow instance based on the SSD Mobilenet V2 Object detection model (https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2) and performs detection. Afterwards the Result is returned to the main thread and displayed with bounding boxes.
The program uses the tensorflwo C API to be lihtweigth and allowing precompiled sources from (############).

Features
- The frame reader thread always tries to fill a frambuffer with the next 5 frames.
- The main loop displays a new frame every 30msec
- The CNN is only instanciated once and then fed conecutively
- As soon as the detector is done, a new frame is moved to the detcor thread instance (hence not displayed)
- The detector results are displayed in a secondary window


# Installation

# Know Issues
- Itried to replace the pointers and mallocs from the tensofwlo
