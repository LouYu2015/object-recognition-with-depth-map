SET(execName ZED_Project)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(${execName})
 
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 OLD)
    cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)
 
SET(EXECUTABLE_OUTPUT_PATH ".")
 
SET(VERSION_REQ_CUDA "8.0")
 
IF(WIN32) # Windows
    if (CMAKE_CL_64) 
        SET(ZED_INCLUDE_DIRS $ENV{ZED_INCLUDE_DIRS})
        SET(ZED_LIBRARIES $ENV{ZED_LIBRARIES_64})
        SET(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})
    else()
        message(FATAL_ERROR "You've selected the 32bit version of ${CMAKE_GENERATOR}. \n Please delete the cache (file->Delete Cache) and use the 64bit version. (${CMAKE_GENERATOR} Win64)")
    endif()
ELSE() # Linux
    find_package(ZED 2 REQUIRED)
    SET(SPECIAL_OS_LIBS "pthread")
ENDIF(WIN32)
 
find_package(CUDA ${VERSION_REQ_CUDA} REQUIRED)
  
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
 
link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
 
SET(SRC_FOLDER src)
FILE(GLOB_RECURSE SRC_FILES "${SRC_FOLDER}/*.cpp")  
ADD_EXECUTABLE(${execName} "./main.cpp")
 
add_definitions(-std=c++0x -g -O3)

# Include opencv
if(WIN32)
  set(OpenCV_DIR "d:/libs/opencv-2.4.8/build")
else()
  set(OpenCV_DIR "/usr/lib/opencv")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

target_link_libraries(${execName} ${OpenCV_LIBS})
 
# Add the required libraries for linking:
TARGET_LINK_LIBRARIES(${execName}
                        ${ZED_LIBRARIES}
                        ${SPECIAL_OS_LIBS}
                        ${CUDA_LIBRARIES} ${CUDA_npps_LIBRARY})

# SET(CMAKE_CXX_FLAGS "`pkg-config --cflags --libs opencv`" )
