# YOLO Android ncnn Demo

这个项目是一个使用 Kotlin + Jetpack Compose 编写的 Android YOLO 推理 demo。当前已经接入 ncnn，可以在 Android 设备上用 CPU 或 Vulkan GPU 运行 YOLO 检测模型，并支持图片推理和摄像头实时推理。

> 说明：项目目标中提到的 ONNX Runtime 还没有接入；当前可用的推理后端是 ncnn。

## 功能

- 图片推理：从系统相册选择图片，运行 YOLO 检测并显示结果图。
- 摄像头实时推理：使用 CameraX 获取后置摄像头画面，实时检测并绘制检测框。
- CPU / GPU 切换：默认使用 GPU，界面上可以手动切换 CPU 或 GPU。
- GPU 失败回退：如果 GPU 初始化或推理失败，会提示错误并自动切换到 CPU。
- 开始 / 暂停摄像头推理：摄像头模式下可以暂停或恢复分析帧。
- 置信度和 IoU 阈值设置：界面提供可折叠的阈值设置区域。
- 结果可视化：检测框左上角显示类别名和置信度。
- 性能显示：摄像头实时推理时显示单帧推理耗时和 FPS。
- 模型配置读取：输入尺寸、类别数量和类别名称从 `metadata.yaml` 读取，不再写死在 C++ 中。
- App 图标：使用根目录 `logo.png` 生成的 adaptive launcher icon。

## 编程环境

推荐环境：

- Windows 11
- Android Studio
- Android Gradle Plugin 9.2.1
- Kotlin Compose plugin 2.2.10
- Gradle Wrapper
- Android SDK compileSdk 36.1
- Android minSdk 27
- JDK：使用 Android Studio 自带 JBR
- Android NDK + CMake
- ncnn Android Vulkan SDK：`ncnn-20260526-android-vulkan`

## Android 依赖

主要 Android/Kotlin 依赖在 `gradle/libs.versions.toml` 和 `app/build.gradle.kts` 中维护：

- Jetpack Compose BOM `2026.02.01`
- Material 3
- Activity Compose
- Core KTX
- Lifecycle Runtime KTX
- CameraX `1.5.1`
  - `camera-core`
  - `camera-camera2`
  - `camera-lifecycle`
- JUnit / AndroidX Test / Espresso

Native 侧依赖：

- C++17
- Android `jnigraphics`
- Android `log`
- ncnn
- Vulkan GPU 推理能力由 ncnn Android Vulkan 包提供

当前构建 ABI：

- `arm64-v8a`
- `armeabi-v7a`

## ncnn SDK 放置方式

当前仓库不内置 ncnn Android SDK。请下载 ncnn Android Vulkan 预编译包，解压后放到：

```text
app/src/main/cpp/ncnn-android-vulkan/
```

期望目录结构类似：

```text
app/src/main/cpp/ncnn-android-vulkan/
  arm64-v8a/
    include/
    lib/cmake/ncnn/ncnnConfig.cmake
  armeabi-v7a/
    include/
    lib/cmake/ncnn/ncnnConfig.cmake
```

如果这个目录不存在，CMake 会编译 JNI stub，App 可以启动，但推理时会提示需要配置 ncnn SDK。

## 模型文件

当前模型资产目录：

```text
app/src/main/assets/yolo26n_ncnn_model/
  model.ncnn.param
  model.ncnn.bin
  metadata.yaml
```

`metadata.yaml` 中至少需要包含：

- `imgsz`：模型输入尺寸，例如 `[640, 640]`
- `names`：类别 id 到类别名的映射

Kotlin JNI 包装类会把这些资产路径传给 C++：

```text
app/src/main/java/com/example/yolo_app/detector/NcnnYoloDetector.kt
```

C++ ncnn 实现位于：

```text
app/src/main/cpp/ncnn_yolo_jni.cpp
```

## 推理流程

1. Kotlin UI 获取图片或 CameraX 视频帧。
2. 根据 EXIF 或 CameraX rotation 信息修正图像方向。
3. 调用 `NcnnYoloDetector.detect(...)`。
4. Native C++ 将 Android `Bitmap` 转为 ncnn 输入。
5. 按 `metadata.yaml` 读取到的输入尺寸进行 letterbox 预处理。
6. 调用 ncnn 网络推理。
7. 解析 YOLO 输出，执行置信度过滤和 NMS。
8. 返回 Kotlin `Detection` 列表。
9. Compose Canvas 绘制检测框、类别、分数、耗时和 FPS。

模型不是每次检测都重新载入。App 会按当前设备类型维护 detector 实例，第一次推理时创建并加载模型，之后复用；切换 CPU/GPU 或释放资源时才会销毁并重建。

## 构建

在项目根目录执行：

```powershell
$env:hTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
$env:JAVA_HOME="C:\Program Files\Android\Android Studio\jbr"
$env:Path="$env:JAVA_HOME\bin;$env:Path"
.\gradlew.bat assembleDebug
```

构建成功后 APK 位于：

```text
app/build/outputs/apk/debug/app-debug.apk
```

## 安装到设备

连接 Android 设备并确认 adb 可见：

```powershell
adb devices
```

安装 debug APK：

```powershell
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

启动应用：

```powershell
adb shell am start -n com.example.yolo_app/.MainActivity
```

如果摄像头权限没有自动弹出或需要手动授权：

```powershell
adb shell pm grant com.example.yolo_app android.permission.CAMERA
```

## 主要文件

```text
app/src/main/java/com/example/yolo_app/MainActivity.kt
app/src/main/java/com/example/yolo_app/detector/NcnnYoloDetector.kt
app/src/main/java/com/example/yolo_app/detector/Detection.kt
app/src/main/cpp/ncnn_yolo_jni.cpp
app/src/main/cpp/ncnn_yolo_stub_jni.cpp
app/src/main/cpp/CMakeLists.txt
app/src/main/assets/yolo26n_ncnn_model/
```

## 当前限制

- 当前只实现 ncnn 后端，ONNX Runtime 还没有集成。
- Native 解析逻辑按当前 YOLO ncnn 输出格式实现，替换不同模型时需要确认输出张量布局是否一致。
- `metadata.yaml` 解析是轻量实现，适合当前 Ultralytics 导出的 metadata 格式。
- 摄像头实时推理当前使用后置摄像头。
