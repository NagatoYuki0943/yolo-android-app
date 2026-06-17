#include <jni.h>

namespace {

void throwIllegalState(JNIEnv* env, const char* message) {
    jclass clazz = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(clazz, message);
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_yolo_1app_detector_NcnnYoloDetector_nativeCreate(
        JNIEnv* env,
        jobject,
        jobject,
        jstring,
        jstring,
        jstring,
        jboolean) {
    throwIllegalState(
            env,
            "ncnn Android SDK is not configured. Extract ncnn Android Vulkan SDK to app/src/main/cpp/ncnn-android-vulkan.");
    return 0;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_example_yolo_1app_detector_NcnnYoloDetector_nativeDetect(
        JNIEnv* env,
        jobject,
        jlong,
        jobject,
        jfloat,
        jfloat) {
    throwIllegalState(env, "ncnn detector is not available.");
    jclass detectionClass = env->FindClass("com/example/yolo_app/detector/Detection");
    return env->NewObjectArray(0, detectionClass, nullptr);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_yolo_1app_detector_NcnnYoloDetector_nativeDestroy(JNIEnv*, jobject, jlong) {}
