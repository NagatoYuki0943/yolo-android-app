#include <algorithm>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <jni.h>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

#include "net.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif

extern "C" void omp_set_num_threads(int num_threads) __attribute__((weak));
extern "C" void omp_set_dynamic(int dynamic) __attribute__((weak));
extern "C" void kmp_set_blocktime(int blocktime) __attribute__((weak));
extern "C" void kmp_set_defaults(char const* defaults) __attribute__((weak));

namespace {

struct Object {
    float left;
    float top;
    float right;
    float bottom;
    int label;
    float prob;
};

struct Detector {
    ncnn::Net net;
    int inputWidth = 0;
    int inputHeight = 0;
    std::vector<std::string> labels;
};

void configureOpenMpRuntimeOnce() {
    static std::once_flag onceFlag;
    std::call_once(onceFlag, []() {
        setenv("KMP_AFFINITY", "none", 1);
        setenv("OMP_PROC_BIND", "FALSE", 1);
        setenv("OMP_NUM_THREADS", "1", 1);
        if (kmp_set_defaults != nullptr) {
            kmp_set_defaults("KMP_AFFINITY=none");
        }
        if (omp_set_dynamic != nullptr) {
            omp_set_dynamic(0);
        }
        if (omp_set_num_threads != nullptr) {
            omp_set_num_threads(1);
        }
        if (kmp_set_blocktime != nullptr) {
            kmp_set_blocktime(0);
        }
    });
}

void throwException(JNIEnv* env, const char* className, const std::string& message) {
    jclass clazz = env->FindClass(className);
    env->ThrowNew(clazz, message.c_str());
}

std::string trim(const std::string& value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        begin++;
    }
    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        end--;
    }
    return value.substr(begin, end - begin);
}

bool startsWith(const std::string& value, const char* prefix) {
    return value.rfind(prefix, 0) == 0;
}

bool parseInt(const std::string& value, int* output) {
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str()) {
        return false;
    }
    *output = static_cast<int>(parsed);
    return true;
}

std::string readAssetText(AAssetManager* mgr, const char* assetPath) {
    AAsset* asset = AAssetManager_open(mgr, assetPath, AASSET_MODE_BUFFER);
    if (asset == nullptr) {
        return {};
    }

    const off_t length = AAsset_getLength(asset);
    std::string content;
    content.resize(static_cast<size_t>(length));
    if (length > 0) {
        AAsset_read(asset, content.data(), static_cast<size_t>(length));
    }
    AAsset_close(asset);
    return content;
}

std::string unquote(const std::string& value) {
    if (value.size() >= 2) {
        const char first = value.front();
        const char last = value.back();
        if ((first == '\'' && last == '\'') || (first == '"' && last == '"')) {
            return value.substr(1, value.size() - 2);
        }
    }
    return value;
}

void parseMetadataYaml(const std::string& yaml, Detector& detector) {
    bool inImageSize = false;
    bool inNames = false;
    int imageSizeValues[2] = {0, 0};
    int imageSizeCount = 0;

    size_t lineStart = 0;
    while (lineStart <= yaml.size()) {
        size_t lineEnd = yaml.find('\n', lineStart);
        if (lineEnd == std::string::npos) {
            lineEnd = yaml.size();
        }

        std::string line = yaml.substr(lineStart, lineEnd - lineStart);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::string stripped = trim(line);

        if (stripped == "imgsz:") {
            inImageSize = true;
            inNames = false;
        } else if (stripped == "names:") {
            inImageSize = false;
            inNames = true;
        } else if (!stripped.empty() && stripped.back() == ':' && line.empty()) {
            inImageSize = false;
            inNames = false;
        } else if (!stripped.empty() && stripped[0] != '-' && line[0] != ' ' && line[0] != '\t' && stripped.find(':') != std::string::npos) {
            inImageSize = false;
            inNames = false;
        } else if (inImageSize && startsWith(stripped, "- ")) {
            int parsed = 0;
            if (imageSizeCount < 2 && parseInt(trim(stripped.substr(2)), &parsed) && parsed > 0) {
                imageSizeValues[imageSizeCount++] = parsed;
            }
        } else if (inNames) {
            const size_t colon = stripped.find(':');
            if (colon != std::string::npos) {
                int classId = 0;
                if (parseInt(trim(stripped.substr(0, colon)), &classId) && classId >= 0) {
                    if (static_cast<size_t>(classId) >= detector.labels.size()) {
                        detector.labels.resize(static_cast<size_t>(classId) + 1);
                    }
                    detector.labels[static_cast<size_t>(classId)] = unquote(trim(stripped.substr(colon + 1)));
                }
            }
        }

        if (lineEnd == yaml.size()) {
            break;
        }
        lineStart = lineEnd + 1;
    }

    if (imageSizeCount >= 1) {
        detector.inputWidth = imageSizeValues[0];
        detector.inputHeight = imageSizeCount >= 2 ? imageSizeValues[1] : imageSizeValues[0];
    }
}

float intersectionArea(const Object& a, const Object& b) {
    const float left = std::max(a.left, b.left);
    const float top = std::max(a.top, b.top);
    const float right = std::min(a.right, b.right);
    const float bottom = std::min(a.bottom, b.bottom);
    return std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
}

void nmsSortedBboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nmsThreshold) {
    picked.clear();
    std::vector<float> areas(objects.size());
    for (size_t i = 0; i < objects.size(); i++) {
        areas[i] = (objects[i].right - objects[i].left) * (objects[i].bottom - objects[i].top);
    }

    for (size_t i = 0; i < objects.size(); i++) {
        const Object& candidate = objects[i];
        bool keep = true;
        for (int pickedIndex : picked) {
            const Object& selected = objects[pickedIndex];
            const float interArea = intersectionArea(candidate, selected);
            const float unionArea = areas[i] + areas[pickedIndex] - interArea;
            if (unionArea > 0.0f && interArea / unionArea > nmsThreshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            picked.push_back(static_cast<int>(i));
        }
    }
}

std::vector<Object> parseOutput(
        const Detector& detector,
        const ncnn::Mat& out,
        int imageWidth,
        int imageHeight,
        float scoreThreshold,
        float nmsThreshold) {
    std::vector<Object> proposals;
    const int classCount = static_cast<int>(detector.labels.size());
    const int outputChannels = 4 + classCount;
    const float scale = std::min(static_cast<float>(detector.inputWidth) / imageWidth, static_cast<float>(detector.inputHeight) / imageHeight);
    const float padX = (detector.inputWidth - imageWidth * scale) * 0.5f;
    const float padY = (detector.inputHeight - imageHeight * scale) * 0.5f;

    const int featureCount = out.h == outputChannels ? out.w : out.h;
    const bool channelFirst = out.h == outputChannels;

    for (int i = 0; i < featureCount; i++) {
        float cx;
        float cy;
        float w;
        float h;
        int label = -1;
        float score = 0.0f;

        if (channelFirst) {
            cx = out.row(0)[i];
            cy = out.row(1)[i];
            w = out.row(2)[i];
            h = out.row(3)[i];
            for (int classId = 0; classId < classCount; classId++) {
                const float classScore = out.row(4 + classId)[i];
                if (classScore > score) {
                    score = classScore;
                    label = classId;
                }
            }
        } else {
            const float* row = out.row(i);
            cx = row[0];
            cy = row[1];
            w = row[2];
            h = row[3];
            for (int classId = 0; classId < classCount; classId++) {
                const float classScore = row[4 + classId];
                if (classScore > score) {
                    score = classScore;
                    label = classId;
                }
            }
        }

        if (score < scoreThreshold) {
            continue;
        }

        Object obj{};
        obj.left = std::clamp((cx - w * 0.5f - padX) / scale, 0.0f, static_cast<float>(imageWidth));
        obj.top = std::clamp((cy - h * 0.5f - padY) / scale, 0.0f, static_cast<float>(imageHeight));
        obj.right = std::clamp((cx + w * 0.5f - padX) / scale, 0.0f, static_cast<float>(imageWidth));
        obj.bottom = std::clamp((cy + h * 0.5f - padY) / scale, 0.0f, static_cast<float>(imageHeight));
        obj.label = label;
        obj.prob = score;
        proposals.push_back(obj);
    }

    std::sort(proposals.begin(), proposals.end(), [](const Object& a, const Object& b) {
        return a.prob > b.prob;
    });

    std::vector<int> picked;
    nmsSortedBboxes(proposals, picked, nmsThreshold);

    std::vector<Object> objects;
    objects.reserve(picked.size());
    for (int index : picked) {
        objects.push_back(proposals[index]);
    }
    return objects;
}

jobjectArray toDetectionArray(JNIEnv* env, const Detector& detector, const std::vector<Object>& objects) {
    jclass detectionClass = env->FindClass("com/example/yolo_app/detector/Detection");
    jmethodID constructor = env->GetMethodID(detectionClass, "<init>", "(Ljava/lang/String;IFFFFF)V");
    jobjectArray array = env->NewObjectArray(static_cast<jsize>(objects.size()), detectionClass, nullptr);

    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];
        const std::string labelText =
                obj.label >= 0 && static_cast<size_t>(obj.label) < detector.labels.size() && !detector.labels[static_cast<size_t>(obj.label)].empty()
                        ? detector.labels[static_cast<size_t>(obj.label)]
                        : std::string("class_") + std::to_string(obj.label);
        jstring label = env->NewStringUTF(labelText.c_str());
        jobject detection = env->NewObject(
                detectionClass,
                constructor,
                label,
                obj.label,
                obj.prob,
                obj.left,
                obj.top,
                obj.right,
                obj.bottom);
        env->SetObjectArrayElement(array, static_cast<jsize>(i), detection);
        env->DeleteLocalRef(label);
        env->DeleteLocalRef(detection);
    }

    return array;
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_yolo_1app_detector_NcnnYoloDetector_nativeCreate(
        JNIEnv* env,
        jobject,
        jobject assetManager,
        jstring paramAssetPath,
        jstring binAssetPath,
        jstring metadataAssetPath,
        jboolean useVulkan) {
    configureOpenMpRuntimeOnce();

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (mgr == nullptr) {
        throwException(env, "java/lang/IllegalArgumentException", "AssetManager is null");
        return 0;
    }

    const char* paramPath = env->GetStringUTFChars(paramAssetPath, nullptr);
    const char* binPath = env->GetStringUTFChars(binAssetPath, nullptr);
    const char* metadataPath = env->GetStringUTFChars(metadataAssetPath, nullptr);

    std::unique_ptr<Detector> detector = std::make_unique<Detector>();
    parseMetadataYaml(readAssetText(mgr, metadataPath), *detector);
    if (detector->inputWidth <= 0 || detector->inputHeight <= 0) {
        env->ReleaseStringUTFChars(paramAssetPath, paramPath);
        env->ReleaseStringUTFChars(binAssetPath, binPath);
        env->ReleaseStringUTFChars(metadataAssetPath, metadataPath);
        throwException(env, "java/lang/IllegalStateException", "Failed to read input size from ncnn metadata.yaml");
        return 0;
    }
    if (detector->labels.empty()) {
        env->ReleaseStringUTFChars(paramAssetPath, paramPath);
        env->ReleaseStringUTFChars(binAssetPath, binPath);
        env->ReleaseStringUTFChars(metadataAssetPath, metadataPath);
        throwException(env, "java/lang/IllegalStateException", "Failed to read class names from ncnn metadata.yaml");
        return 0;
    }

    detector->net.opt.use_fp16_packed = true;
    detector->net.opt.use_fp16_storage = true;
    detector->net.opt.use_fp16_arithmetic = false;
    detector->net.opt.num_threads = 4;
    detector->net.opt.openmp_blocktime = 0;
#if NCNN_VULKAN
    detector->net.opt.use_vulkan_compute = useVulkan == JNI_TRUE;
#else
    (void)useVulkan;
#endif

    const int paramResult = detector->net.load_param(mgr, paramPath);
    const int modelResult = detector->net.load_model(mgr, binPath);

    env->ReleaseStringUTFChars(paramAssetPath, paramPath);
    env->ReleaseStringUTFChars(binAssetPath, binPath);
    env->ReleaseStringUTFChars(metadataAssetPath, metadataPath);

    if (paramResult != 0 || modelResult != 0) {
        throwException(env, "java/lang/IllegalStateException", "Failed to load ncnn YOLO model from assets");
        return 0;
    }

    return reinterpret_cast<jlong>(detector.release());
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_example_yolo_1app_detector_NcnnYoloDetector_nativeDetect(
        JNIEnv* env,
        jobject,
        jlong nativeHandle,
        jobject bitmap,
        jfloat scoreThreshold,
        jfloat nmsThreshold) {
    auto* detector = reinterpret_cast<Detector*>(nativeHandle);
    if (detector == nullptr) {
        throwException(env, "java/lang/IllegalStateException", "Detector is not initialized");
        return nullptr;
    }

    AndroidBitmapInfo bitmapInfo{};
    if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) != ANDROID_BITMAP_RESULT_SUCCESS ||
        bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        throwException(env, "java/lang/IllegalArgumentException", "Bitmap must be ARGB_8888/RGBA_8888");
        return nullptr;
    }

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        throwException(env, "java/lang/IllegalStateException", "Failed to lock bitmap pixels");
        return nullptr;
    }

    const int imageWidth = static_cast<int>(bitmapInfo.width);
    const int imageHeight = static_cast<int>(bitmapInfo.height);
    const float scale = std::min(static_cast<float>(detector->inputWidth) / imageWidth, static_cast<float>(detector->inputHeight) / imageHeight);
    const int resizedWidth = static_cast<int>(std::round(imageWidth * scale));
    const int resizedHeight = static_cast<int>(std::round(imageHeight * scale));
    const int padLeft = (detector->inputWidth - resizedWidth) / 2;
    const int padRight = detector->inputWidth - resizedWidth - padLeft;
    const int padTop = (detector->inputHeight - resizedHeight) / 2;
    const int padBottom = detector->inputHeight - resizedHeight - padTop;

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(
            static_cast<const unsigned char*>(pixels),
            ncnn::Mat::PIXEL_RGBA2RGB,
            imageWidth,
            imageHeight,
            resizedWidth,
            resizedHeight);

    AndroidBitmap_unlockPixels(env, bitmap);

    ncnn::Mat padded;
    ncnn::copy_make_border(input, padded, padTop, padBottom, padLeft, padRight, ncnn::BORDER_CONSTANT, 114.0f);
    const float norm[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    padded.substract_mean_normalize(nullptr, norm);

    ncnn::Extractor extractor = detector->net.create_extractor();
    extractor.input("in0", padded);

    ncnn::Mat out;
    if (extractor.extract("out0", out) != 0) {
        throwException(env, "java/lang/IllegalStateException", "Failed to extract YOLO output blob out0");
        return nullptr;
    }

    std::vector<Object> objects = parseOutput(*detector, out, imageWidth, imageHeight, scoreThreshold, nmsThreshold);
    return toDetectionArray(env, *detector, objects);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_yolo_1app_detector_NcnnYoloDetector_nativeDestroy(JNIEnv*, jobject, jlong nativeHandle) {
    delete reinterpret_cast<Detector*>(nativeHandle);
}
