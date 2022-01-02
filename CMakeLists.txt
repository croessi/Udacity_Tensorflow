cmake_minimum_required(VERSION 3.0.0)
project(Udacity-Tensorflow VERSION 0.1.0)

set (CMAKE_CXX_STANDARD 17)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(OpenCV 4.1 REQUIRED)

#find_package(PkgConfig REQUIRED)
#pkg_check_modules(deps REQUIRED IMPORTED_TARGET mqtt)

#pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0 gstreamer-1.0)
#target_link_libraries(target )

# Find all executables
file(GLOB project_SRCS src/*.cpp)# src/*.h)

add_executable(Udacity-Tensorflow ${project_SRCS})

#include_directories (/usr/include/gstreamer-1.0 /usr/include/glib-2.0 /lib/x86_64-linux-gnu/glib-2.0/include)

#target_link_libraries(Udacity-Tensorflow ${OpenCV_LIBRARIES} ${X11_LIBRARIES} tensorflow Threads::Threads PkgConfig::deps)
target_link_libraries(Udacity-Tensorflow ${OpenCV_LIBRARIES} ${PAHO_CPP_LIB} ${PAHO_C_LIB} paho-mqtt3c paho-mqtt3a paho-mqttpp3 tensorflow Threads::Threads)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

