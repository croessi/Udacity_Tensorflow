# Udacity_Tensorflow

This program asynchronously reads in video frames via Opencv / FFMPEG and sends it to a second thread with a tensorflow instance based on the SSD Mobilenet V2 Object detection model (https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2) and performs detection. Afterwards the result is returned to the main thread and displayed with bounding boxes.
To be lightweight and allowing precompiled sources the program uses the tensorflow C API from https://www.tensorflow.org/install/lang_c.

![Udacity-Tensorflow](https://user-images.githubusercontent.com/87674139/147240437-e9ddbf63-6ceb-4b1c-b85e-5bf9bb60f0aa.png)

### Features
- The frame reader thread fills a framebuffer asynchronously with the next 3 frames.
- The main loop displays a new frame every 30msec
- The CNN is only instantiated once and then fed consecutively
- As soon as the detector is done, a new frame is moved to the detector thread instance (hence not displayed until detection is through)
- The detector results are displayed in a secondary window
- Easy switching to other pre-trained models from TensorHub possible


# Installation
### Prerequisites
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
- **main.cpp**
  - detects if display is available - if not no CV Windows are created 
  - main loop instantiating the VideoReader, a Detector and a TensorProcessor Object and starts the corresponding threads
  - while loop to
   - read frames
   - creates a new DetectionResult Object as soon as the current detection is done and moves the current frame into the new object which is then moved to the TensorProcessor thread for detection   
    - code to display outputs
 
- **ReadClassesToLabels.cpp/h**
  - Function to parse the class labels from a text file to a map

- **TensorProcessor.cpp/h**
  - **DetectorClass** as base Class for various Detector Models with virtual method GetName
  - **MobilenetV2Class** with specific config for MobilenetV2 Model
  - **Detection_t** holds the structure for a single detected object
  - **DetectionResultClass** holding the image and the detection outputs - object is moveable by implementing the rule of 5
  - **Message queue Class** for the input and output queue for the detector
  - **TensorProcessorClass** requiering a detector object as input and providing methods to start and step the dector thread  
- **VideoReader.cpp/h**
  - **VideoReaderClass** accepting a filename as input - decode is doing with OpenCV//FMPEG
   - Methods to Start and Stop the Grabber thread
   - Grabber fetches 3 frames and stores them in the framebuffer 
   
# Know Issues
- [x] no suitable use case found to use promise and future to pass data from a worker thread to a parent thread
- [ ] pointers and mallocs from tensorflow C API to be replaced with smart pointers and vectors - did not work at the first try

# Rubric points

### Loops, Functions, I/O
- A variety of control structures are used in the project (main.cpp 122)
- The project code is clearly organized into functions (multiple functions in multiple classes)
- The project reads data from an external file or writes data to a file as part of the necessary operation of the program (see ReadClassesToLabels)
- 	The project accepts input from a user as part of the necessary operation of the program (main.cpp 87)

### Object Oriented Programming
- The project code is organized into classes with class attributes to hold the data, and class methods to perform tasks. (TensorProcessor.h / VideoReader.h)
- All class data members are explicitly specified as public, protected, or private. (Tensorprocessor and VideoReaderClass)
- All class members that are set to argument values are initialized through member initialization lists (Tensorprocessor.cpp 28 / Tensorprocessor.h 33)
- All class member functions document their effects, either through function names, comments, or formal documentation. Member functions do not change program state in undocumented ways. (see Unktions/Classes)
- Appropriate data and functions are grouped into classes. Member data that is subject to an invariant is hidden from the user. State is accessed via member functions.(See defined classes)
- Inheritance hierarchies are logical. Composition is used instead of inheritance when appropriate. Abstract classes are composed of pure virtual functions. Override functions are specified. (see defined classes)
- One function is overloaded with different signatures for the same function name. (Constructor overload in TensorProcessor.cpp 28) 
- One member function in an inherited class overrides a virtual base class member function. (TensorProcessor.h 71)
- One function is declared with a template that allows it to accept a generic parameter. (Tensorprocessor.h 166)

### Memory Management
- At least two variables are defined as references, or two functions use pass-by-reference in the project code.(ReadClassestoLabels.h 13 / Tensorprocessor.h 68)
- At least one class that uses unmanaged dynamically allocated memory, along with any class that otherwise needs to modify state upon the termination of an object, uses a destructor. (TensorProcessor.cpp 47,60 & 77,78)
- The project follows the Resource Acquisition Is Initialization pattern where appropriate, by allocating objects at compile-time, initializing objects when they are declared, and utilizing scope to ensure their automatic destruction. (see classes)
- For all classes, if any one of the copy constructor, copy assignment operator, move constructor, move assignment operator, and destructor are defined, then all of these functions are defined. (TensorProcessor.h 109)
- For classes with move constructors, the project returns objects of that class by value, and relies on the move constructor, instead of copying the object.(TensorProcessor.h 111, 141)
- The project uses at least one smart pointer: unique_ptr, shared_ptr, or weak_ptr. The project does not use raw pointers. (main.cpp 63)

### Concurrency
- The project uses multiple threads in the execution. (VideoReader.cpp 6 / TensorProcessor.cpp 16)
- A promise and future is used to pass data from a worker thread to a parent thread in the project code (main.cpp 46)
- A mutex or lock (e.g. std::lock_guard or `std::unique_lock) is used to protect data that is shared across multiple threads in the project code. (VodeoReader.cpp 12)
- A std::condition_variable is used in the project code to synchronize thread execution. (TensorProcessor.h 202)
