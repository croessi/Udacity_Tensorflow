cmake_minimum_required(VERSION 3.0.0)
project(Udacity-Tensorflow VERSION 0.1.0)

set (CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4.1 REQUIRED)
#find_package(TensorFlow REQUIRED)

# Find all executables
file(GLOB project_SRCS src/*.cpp) #src/*.h

add_executable(Udacity-Tensorflow ${project_SRCS})

target_link_libraries(Udacity-Tensorflow ${OpenCV_LIBRARIES} tensorflow)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
