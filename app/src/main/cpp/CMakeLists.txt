cmake_minimum_required(VERSION 3.22.1)

project(yolo_ncnn)

set(NCNN_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn-android-vulkan" CACHE PATH "Path to the extracted ncnn Android Vulkan SDK")
set(NCNN_CMAKE_DIR "${NCNN_SDK_DIR}/${ANDROID_ABI}/lib/cmake/ncnn")

if(EXISTS "${NCNN_CMAKE_DIR}/ncnnConfig.cmake")
    set(ncnn_DIR "${NCNN_CMAKE_DIR}")
    find_package(ncnn REQUIRED)
    add_library(yolo_ncnn SHARED ncnn_yolo_jni.cpp)
    target_link_libraries(yolo_ncnn ncnn android jnigraphics log)
else()
    message(WARNING "ncnn Android SDK not found at ${NCNN_CMAKE_DIR}; building JNI stub. Put the extracted SDK under app/src/main/cpp/ncnn-android-vulkan or pass -DNCNN_SDK_DIR=...")
    add_library(yolo_ncnn SHARED ncnn_yolo_stub_jni.cpp)
    target_link_libraries(yolo_ncnn android jnigraphics log)
endif()
