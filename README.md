# Udacity_Tensorflow

This program asynchronously reads in video frames via Opencv / FFMPEG and sends it to a second thread with a tensorflow instance based on the SSD Mobilenet V2 Object detection model (https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2) and performs detection. Afterwards the Result is returned to the main thread and displayed with bounding boxes.
The program uses the tensorflwo C API to be lihtweigth and allowing precompiled sources from https://www.tensorflow.org/install/lang_c

![Udacity-Tensorflow](https://user-images.githubusercontent.com/87674139/147240437-e9ddbf63-6ceb-4b1c-b85e-5bf9bb60f0aa.png)

Features
- The frame reader thread always tries to fill a frambuffer with the next 3 frames.
- The main loop displays a new frame every 30msec
- The CNN is only instanciated once and then fed conecutively
- As soon as the detector is done, a new frame is moved to the detcor thread instance (hence not displayed until detrction is through)
- The detector results are displayed in a secondary window
- Easy switch to other pre-trained models from TensorHub


# Installation
### Prerequistes
- Tested on Ubuntu 20.04
```bash
sudo apt install g++
sudo apt-get install cmake
sudo apt install libopencv-dev
```

### Tensorflow for C
```bash
FILENAME=libtensorflow-cpu-linux-x86_64-2.7.0.tar.gz
wget -q --no-check-certificate https://storage.googleapis.com/tensorflow/libtensorflow/${FILENAME}
sudo tar -C /usr/local -xzf ${FILENAME}
sudo ldconfig /usr/local/lib
```

###  Udacity Tensorflow
```bash
 git clone https://github.com/croessi/Udacity_Tensorflow
 cd Udacity_Tensorflow
 mkdir build
 cd build
 cmake ..
 ```
 # File and Class description
### Model Folder
- Model has to be in the ./ssd_mobilenet_v2 folder as Tensorflow v2 Model structure. 
- Additionally the file with the class labels "mscoco_label_map.pbtxt" has to be placed there. 

### Sources
- main.cpp
       - detects if display is availble - if not no CV Windows are created 
       - main loop instanciating the VideoReader, a Detector and a TensorProcessor Object and starts the corresponding threads
       - while loop to
              - read frames
              - creates snew DetectionResult Object as soon as the current detection is done and moves the current frame into the new object which is then moved to the TensorProcessor thread for detection   
       - code to display outputs
 
- ReadClassesToLabels.cpp/h
       - Function to parse the class labels from a text file to a map

- TensorProcessor.cpp/h
       - DetectorClass as base Class for various Detector Models with virtual method GetName
       - MobilenetV2Class with specific config for MobilenetV2 Model
       - Detection_t holds the structure for a single detected object
       - DetectionResultClass holding the image and the detection outputs - object is moveable by implementing the rule of 5
       - Message queue Class for the input and output queue for the detector
       - TensorProcessorClass requiering a detector object as input and providing methods to start and step the dector thread  
- VideoReader.cpp/h
       - VideoReaderClass accepting a filename as input - decode is doing with OpenCV//FMPEG
              - Methods to Start and Stop the Grabber thread
              - Grabber is always tryeing to keep 3 frames in buffer 
# Know Issues
- [] no suitable usa case found to use promise and future to pass data from a worker thread to a parent thread in the project code.
- [] pointers and mallocs from tensorflow C API to be replaced with smart pointers and vectors - did not work in the first try
