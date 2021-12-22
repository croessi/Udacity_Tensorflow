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
### Prerequistes
- Tested on Ubuntu 20.X

### Tensorflow for C
excute on shell:

FILENAME=libtensorflow-cpu-linux-x86_64-2.7.0.tar.gz
wget -q --no-check-certificate https://storage.googleapis.com/tensorflow/libtensorflow/${FILENAME}
sudo tar -C /usr/local -xzf ${FILENAME}

sudo ldconfig /usr/local/lib

### OpenCV with FFMPEG
Prerequistes (copied from here https://linuxize.com/post/how-to-install-opencv-on-ubuntu-20-04/ - folow the instructions there if you do not want to oad pre-build OpenCV):
sudo apt install pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev
sudo apt install gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev libopenexr-dev libgstreamer-plugins-base1.0-dev
sudo apt libgstreamer1.0-dev

Opencv from Package Manager:
sudo apt install libopencv-dev

### OpenCV with FFMPEG

# Know Issues
- Itried to replace the pointers and mallocs from the tensofwlo
