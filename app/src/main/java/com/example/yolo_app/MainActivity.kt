package com.example.yolo_app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.graphics.Paint
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Size as AndroidSize
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.yolo_app.detector.Detection
import com.example.yolo_app.detector.NcnnModelType
import com.example.yolo_app.detector.NcnnYoloDetector
import com.example.yolo_app.detector.OnnxYoloDetector
import com.example.yolo_app.ui.theme.YoloappTheme
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max
import kotlin.math.roundToInt

private const val DefaultColorClassCount = 80

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            YoloappTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    YoloScreen(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

private enum class InferenceMode {
    Image,
    Camera,
}

private enum class InferenceDevice(
    val displayName: String,
    val useGpu: Boolean,
) {
    Gpu("GPU", true),
    Cpu("CPU", false),
}

private fun InferenceDevice.displayNameFor(engine: InferenceEngine): String {
    return if (engine == InferenceEngine.Onnx && this == InferenceDevice.Gpu) {
        "NNAPI"
    } else {
        displayName
    }
}

private enum class InferenceEngine(
    val displayName: String,
) {
    Ncnn("ncnn"),
    Onnx("ONNX Runtime"),
}

private fun NcnnModelType.displaySuffix(engine: InferenceEngine): String {
    return if (engine == InferenceEngine.Ncnn) "$displayName " else ""
}

private data class TimedDetectionResult(
    val detections: List<Detection>,
    val elapsedMs: Float,
    val engine: InferenceEngine,
    val device: InferenceDevice,
    val ncnnModelType: NcnnModelType,
    val fallbackMessage: String? = null,
)

private data class InferenceOptions(
    val engine: InferenceEngine,
    val device: InferenceDevice,
    val ncnnModelType: NcnnModelType,
    val scoreThreshold: Float,
    val iouThreshold: Float,
)

private class DetectorHolder(
    private val context: Context,
) : AutoCloseable {
    private var detector: ActiveDetector? = null
    private var engine: InferenceEngine? = null
    private var device: InferenceDevice? = null
    private var ncnnModelType: NcnnModelType? = null
    private val unavailableGpuMessages = mutableMapOf<InferenceEngine, String>()

    @Synchronized
    fun closeCurrent() {
        detector?.close()
        detector = null
        engine = null
        device = null
        ncnnModelType = null
    }

    @Synchronized
    fun detect(
        bitmap: Bitmap,
        requestedEngine: InferenceEngine,
        requestedDevice: InferenceDevice,
        requestedNcnnModelType: NcnnModelType,
        scoreThreshold: Float,
        iouThreshold: Float,
    ): TimedDetectionResult {
        return tryDetect(
            bitmap = bitmap,
            requestedEngine = requestedEngine,
            requestedDevice = requestedDevice,
            requestedNcnnModelType = requestedNcnnModelType,
            scoreThreshold = scoreThreshold,
            iouThreshold = iouThreshold,
            fallbackMessage = null,
        )
    }

    @Synchronized
    override fun close() {
        closeCurrent()
    }

    private fun tryDetect(
        bitmap: Bitmap,
        requestedEngine: InferenceEngine,
        requestedDevice: InferenceDevice,
        requestedNcnnModelType: NcnnModelType,
        scoreThreshold: Float,
        iouThreshold: Float,
        fallbackMessage: String?,
    ): TimedDetectionResult {
        if (requestedDevice == InferenceDevice.Gpu) {
            val unavailableMessage = unavailableGpuMessages[requestedEngine]
            if (unavailableMessage != null) {
                return tryDetect(
                    bitmap = bitmap,
                    requestedEngine = requestedEngine,
                    requestedDevice = InferenceDevice.Cpu,
                    requestedNcnnModelType = requestedNcnnModelType,
                    scoreThreshold = scoreThreshold,
                    iouThreshold = iouThreshold,
                    fallbackMessage = unavailableMessage,
                )
            }
        }

        return try {
            val activeDetector = ensureDetector(requestedEngine, requestedDevice, requestedNcnnModelType)
            val startNs = SystemClock.elapsedRealtimeNanos()
            val detections = activeDetector.detect(
                bitmap = bitmap,
                scoreThreshold = scoreThreshold,
                nmsThreshold = iouThreshold,
            )
            val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000.0f
            TimedDetectionResult(
                detections = detections,
                elapsedMs = elapsedMs,
                engine = requestedEngine,
                device = requestedDevice,
                ncnnModelType = requestedNcnnModelType,
                fallbackMessage = fallbackMessage,
            )
        } catch (error: Throwable) {
            closeCurrent()
            if (requestedDevice == InferenceDevice.Gpu) {
                val message = gpuFallbackMessage(requestedEngine, error)
                if (requestedEngine == InferenceEngine.Onnx) {
                    unavailableGpuMessages[requestedEngine] = message
                }
                tryDetect(
                    bitmap = bitmap,
                    requestedEngine = requestedEngine,
                    requestedDevice = InferenceDevice.Cpu,
                    requestedNcnnModelType = requestedNcnnModelType,
                    scoreThreshold = scoreThreshold,
                    iouThreshold = iouThreshold,
                    fallbackMessage = message,
                )
            } else {
                throw error
            }
        }
    }

    private fun ensureDetector(
        requestedEngine: InferenceEngine,
        requestedDevice: InferenceDevice,
        requestedNcnnModelType: NcnnModelType,
    ): ActiveDetector {
        val activeDetector = detector
        val ncnnModelMatches = requestedEngine != InferenceEngine.Ncnn || ncnnModelType == requestedNcnnModelType
        if (activeDetector != null && engine == requestedEngine && device == requestedDevice && ncnnModelMatches) {
            return activeDetector
        }

        closeCurrent()
        val createdDetector = when (requestedEngine) {
            InferenceEngine.Ncnn -> ActiveDetector.Ncnn(
                NcnnYoloDetector(
                    context = context,
                    useGpu = requestedDevice.useGpu,
                    modelType = requestedNcnnModelType,
                ),
            )
            InferenceEngine.Onnx -> ActiveDetector.Onnx(
                OnnxYoloDetector(context, useNnapi = requestedDevice.useGpu),
            )
        }
        return createdDetector.also {
            detector = it
            engine = requestedEngine
            device = requestedDevice
            ncnnModelType = if (requestedEngine == InferenceEngine.Ncnn) requestedNcnnModelType else null
        }
    }

    private sealed class ActiveDetector : AutoCloseable {
        abstract fun detect(bitmap: Bitmap, scoreThreshold: Float, nmsThreshold: Float): List<Detection>

        class Ncnn(private val detector: NcnnYoloDetector) : ActiveDetector() {
            override fun detect(bitmap: Bitmap, scoreThreshold: Float, nmsThreshold: Float): List<Detection> {
                return detector.detect(bitmap, scoreThreshold, nmsThreshold)
            }

            override fun close() = detector.close()
        }

        class Onnx(private val detector: OnnxYoloDetector) : ActiveDetector() {
            override fun detect(bitmap: Bitmap, scoreThreshold: Float, nmsThreshold: Float): List<Detection> {
                return detector.detect(bitmap, scoreThreshold, nmsThreshold)
            }

            override fun close() = detector.close()
        }
    }
}

private fun gpuFallbackMessage(engine: InferenceEngine, error: Throwable): String {
    val detail = error.message ?: error.javaClass.simpleName
    if (engine == InferenceEngine.Onnx) {
        return if (detail.contains("AddNnapiSplit", ignoreCase = true)) {
            "ONNX Runtime NNAPI 不支持当前模型中的 Split 节点，已切换到 CPU"
        } else {
            "ONNX Runtime NNAPI 加速不可用，已切换到 CPU：${detail.take(80)}"
        }
    }
    return "${engine.displayName} GPU 推理失败，已切换到 CPU：${detail.take(80)}"
}

@Composable
fun YoloScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var mode by remember { mutableStateOf(InferenceMode.Image) }
    var imageBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var cameraBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var detections by remember { mutableStateOf<List<Detection>>(emptyList()) }
    var status by remember { mutableStateOf("选择图片或切换到摄像头实时推理") }
    var latencyMs by remember { mutableStateOf<Float?>(null) }
    var fps by remember { mutableStateOf<Float?>(null) }
    var cameraRunning by remember { mutableStateOf(false) }
    var inferenceEngine by remember { mutableStateOf(InferenceEngine.Ncnn) }
    var inferenceDevice by remember { mutableStateOf(InferenceDevice.Gpu) }
    var ncnnModelType by remember { mutableStateOf(NcnnModelType.Float32) }
    var scoreThreshold by remember { mutableStateOf(0.35f) }
    var iouThreshold by remember { mutableStateOf(0.45f) }
    var showThresholds by remember { mutableStateOf(false) }
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED,
        )
    }
    val detectorHolder = remember(context) { DetectorHolder(context.applicationContext) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    val imagePicker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
        uri ?: return@rememberLauncherForActivityResult
        mode = InferenceMode.Image
        imageBitmap = decodeBitmap(context, uri)?.copy(Bitmap.Config.ARGB_8888, false)
        cameraBitmap = null
        detections = emptyList()
        latencyMs = null
        fps = null
        status = if (imageBitmap == null) "图片载入失败" else "图片已载入"
    }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        hasCameraPermission = granted
        cameraRunning = false
        status = if (granted) "摄像头权限已授予，请手动开始摄像头推理" else "需要摄像头权限才能实时推理"
    }

    DisposableEffect(Unit) {
        onDispose {
            detectorHolder.close()
            cameraExecutor.shutdown()
        }
    }

    if (mode == InferenceMode.Camera && hasCameraPermission) {
        CameraInferenceEffect(
            context = context,
            cameraExecutor = cameraExecutor,
            detectorProvider = {
                detectorHolder
            },
            enabled = cameraRunning,
            options = InferenceOptions(
                engine = inferenceEngine,
                device = inferenceDevice,
                ncnnModelType = ncnnModelType,
                scoreThreshold = scoreThreshold,
                iouThreshold = iouThreshold,
            ),
            onFrame = { frame, result ->
                cameraBitmap = frame
                detections = result.detections
                latencyMs = result.elapsedMs
                fps = if (result.elapsedMs > 0.0f) 1000.0f / result.elapsedMs else null
                if (result.fallbackMessage != null && result.device != inferenceDevice) {
                    inferenceDevice = result.device
                }
                status = result.fallbackMessage
                    ?: "实时推理中：${result.detections.size} 个目标，${result.engine.displayName} ${result.ncnnModelType.displaySuffix(result.engine)}${result.device.displayNameFor(result.engine)}"
            },
            onError = { error ->
                status = error.message ?: error.javaClass.simpleName
            },
        )
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(text = "YOLO Demo", style = MaterialTheme.typography.headlineSmall)
        Text(text = status, style = MaterialTheme.typography.bodyMedium)

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ModeButton(
                text = "图片推理",
                selected = mode == InferenceMode.Image,
                onClick = {
                    mode = InferenceMode.Image
                    cameraRunning = false
                    cameraBitmap = null
                    detections = emptyList()
                    latencyMs = null
                    fps = null
                    status = "图片模式"
                },
            )
            ModeButton(
                text = "摄像头推理",
                selected = mode == InferenceMode.Camera,
                onClick = {
                    mode = InferenceMode.Camera
                    imageBitmap = null
                    detections = emptyList()
                    latencyMs = null
                    fps = null
                    if (!hasCameraPermission) {
                        cameraRunning = false
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    } else {
                        cameraRunning = false
                        status = "摄像头推理已关闭，请手动开始"
                    }
                },
            )
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            InferenceEngine.entries.forEach { engine ->
                ModeButton(
                    text = engine.displayName,
                    selected = inferenceEngine == engine,
                    onClick = {
                        if (inferenceEngine != engine) {
                            inferenceEngine = engine
                            if (mode != InferenceMode.Camera) {
                                detectorHolder.closeCurrent()
                            }
                            latencyMs = null
                            fps = null
                            status = if (mode == InferenceMode.Camera) {
                                "将从下一帧切换到 ${engine.displayName}"
                            } else {
                                "已切换到 ${engine.displayName}"
                            }
                        }
                    },
                )
            }
        }

        if (inferenceEngine == InferenceEngine.Ncnn) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                NcnnModelType.entries.forEach { modelType ->
                    ModeButton(
                        text = modelType.displayName,
                        selected = ncnnModelType == modelType,
                        onClick = {
                            if (ncnnModelType != modelType) {
                                ncnnModelType = modelType
                                if (mode != InferenceMode.Camera) {
                                    detectorHolder.closeCurrent()
                                }
                                latencyMs = null
                                fps = null
                                status = if (mode == InferenceMode.Camera) {
                                    "将从下一帧切换到 ncnn ${modelType.displayName}"
                                } else {
                                    "已切换到 ncnn ${modelType.displayName}"
                                }
                            }
                        },
                    )
                }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            InferenceDevice.entries.forEach { device ->
                ModeButton(
                    text = device.displayNameFor(inferenceEngine),
                    selected = inferenceDevice == device,
                    onClick = {
                        if (inferenceDevice != device) {
                            inferenceDevice = device
                            if (mode != InferenceMode.Camera) {
                                detectorHolder.closeCurrent()
                            }
                            latencyMs = null
                            fps = null
                            status = if (mode == InferenceMode.Camera) {
                                "将从下一帧切换到 ${device.displayNameFor(inferenceEngine)} 推理"
                            } else {
                                "已切换到 ${device.displayNameFor(inferenceEngine)} 推理"
                            }
                        }
                    },
                )
            }
        }

        OutlinedButton(
            onClick = { showThresholds = !showThresholds },
        ) {
            Text(text = if (showThresholds) "隐藏参数" else "参数设置")
        }
        Text(
            text = "置信度 ${"%.2f".format(scoreThreshold)} / IoU ${"%.2f".format(iouThreshold)}",
            style = MaterialTheme.typography.bodyMedium,
        )

        if (showThresholds) {
            ThresholdSlider(
                label = "置信度",
                value = scoreThreshold,
                onValueChange = { scoreThreshold = it },
            )
            ThresholdSlider(
                label = "IoU",
                value = iouThreshold,
                onValueChange = { iouThreshold = it },
            )
        }

        if (mode == InferenceMode.Camera) {
            Button(
                onClick = {
                    if (!hasCameraPermission) {
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    } else {
                        cameraRunning = !cameraRunning
                        status = if (cameraRunning) {
                            "正在启动摄像头"
                        } else {
                            "摄像头推理已暂停"
                        }
                    }
                },
            ) {
                Text(if (cameraRunning) "暂停摄像头推理" else "开始摄像头推理")
            }
        } else {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = {
                        imagePicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                    },
                ) {
                    Text("选择图片")
                }

                Button(
                    enabled = imageBitmap != null,
                    onClick = {
                        val source = imageBitmap
                        if (source != null) {
                            try {
                                val result = detectorHolder.detect(
                                    bitmap = source,
                                    requestedEngine = inferenceEngine,
                                    requestedDevice = inferenceDevice,
                                    requestedNcnnModelType = ncnnModelType,
                                    scoreThreshold = scoreThreshold,
                                    iouThreshold = iouThreshold,
                                )
                                detections = result.detections
                                latencyMs = result.elapsedMs
                                fps = null
                                if (result.fallbackMessage != null && result.device != inferenceDevice) {
                                    inferenceDevice = result.device
                                }
                                status = result.fallbackMessage
                                    ?: "图片检测完成：${result.detections.size} 个目标，${result.engine.displayName} ${result.ncnnModelType.displaySuffix(result.engine)}${result.device.displayNameFor(result.engine)}，${"%.1f".format(result.elapsedMs)} ms"
                            } catch (error: Throwable) {
                                status = error.message ?: error.javaClass.simpleName
                            }
                        }
                    },
                ) {
                    Text("运行图片推理")
                }
            }
        }

        val previewBitmap = if (mode == InferenceMode.Camera) cameraBitmap else imageBitmap
        InferenceMetricsRow(
            latencyMs = latencyMs,
            fps = fps,
        )
        DetectionPreview(
            bitmap = previewBitmap,
            detections = detections,
            modifier = previewBitmap.previewModifier(),
        )

        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            detections.forEach { detection ->
                Text(
                    text = "${detection.label} ${"%.1f".format(detection.score * 100)}%  " +
                        "[${detection.left.toInt()}, ${detection.top.toInt()}, ${detection.right.toInt()}, ${detection.bottom.toInt()}]",
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
    }
}

private fun Bitmap?.previewModifier(): Modifier {
    return if (this == null) {
        Modifier
            .fillMaxWidth()
            .heightIn(min = 300.dp, max = 520.dp)
    } else {
        Modifier
            .fillMaxWidth()
            .aspectRatio(width.toFloat() / height.toFloat())
    }
}

@Composable
private fun ModeButton(
    text: String,
    selected: Boolean,
    onClick: () -> Unit,
) {
    if (selected) {
        Button(onClick = onClick) {
            Text(text)
        }
    } else {
        OutlinedButton(onClick = onClick) {
            Text(text)
        }
    }
}

@Composable
private fun ThresholdSlider(
    label: String,
    value: Float,
    onValueChange: (Float) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(
            text = "$label ${"%.2f".format(value)}",
            style = MaterialTheme.typography.bodyMedium,
        )
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = 0.05f..0.95f,
            steps = 17,
        )
    }
}

@Composable
private fun CameraInferenceEffect(
    context: Context,
    cameraExecutor: ExecutorService,
    detectorProvider: () -> DetectorHolder,
    enabled: Boolean,
    options: InferenceOptions,
    onFrame: (Bitmap, TimedDetectionResult) -> Unit,
    onError: (Throwable) -> Unit,
) {
    val latestEnabled = remember { AtomicBoolean(enabled) }
    val latestOptions = remember { AtomicReference(options) }
    val latestOnFrame = remember { AtomicReference(onFrame) }
    val latestOnError = remember { AtomicReference(onError) }
    latestEnabled.set(enabled)
    latestOptions.set(options)
    latestOnFrame.set(onFrame)
    latestOnError.set(onError)

    DisposableEffect(context) {
        val lifecycleOwner = context as LifecycleOwner
        val mainHandler = Handler(Looper.getMainLooper())
        val isActive = AtomicBoolean(true)
        val isDetecting = AtomicBoolean(false)
        val analysisRef = AtomicReference<ImageAnalysis?>(null)
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener(
            {
                if (!isActive.get()) {
                    return@addListener
                }
                try {
                    val cameraProvider = cameraProviderFuture.get()
                    val analysis = ImageAnalysis.Builder()
                        .setTargetResolution(AndroidSize(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()
                    analysisRef.set(analysis)

                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        if (!isActive.get() || !latestEnabled.get() || !isDetecting.compareAndSet(false, true)) {
                            imageProxy.close()
                            return@setAnalyzer
                        }
                        try {
                            val bitmap = imageProxy.toBitmapArgb8888()
                            val currentOptions = latestOptions.get()

                            val result = detectorProvider().detect(
                                bitmap = bitmap,
                                requestedEngine = currentOptions.engine,
                                requestedDevice = currentOptions.device,
                                requestedNcnnModelType = currentOptions.ncnnModelType,
                                scoreThreshold = currentOptions.scoreThreshold,
                                iouThreshold = currentOptions.iouThreshold,
                            )
                            mainHandler.post {
                                if (isActive.get()) {
                                    latestOnFrame.get()(bitmap, result)
                                }
                            }
                        } catch (error: Throwable) {
                            mainHandler.post {
                                if (isActive.get()) {
                                    latestOnError.get()(error)
                                }
                            }
                        } finally {
                            imageProxy.close()
                            isDetecting.set(false)
                        }
                    }

                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        analysis,
                    )
                } catch (error: Throwable) {
                    mainHandler.post {
                        if (isActive.get()) {
                            latestOnError.get()(error)
                        }
                    }
                }
            },
            ContextCompat.getMainExecutor(context),
        )

        onDispose {
            isActive.set(false)
            analysisRef.get()?.clearAnalyzer()
            if (cameraProviderFuture.isDone) {
                cameraProviderFuture.get().unbindAll()
            }
        }
    }
}

@Composable
private fun InferenceMetricsRow(
    latencyMs: Float?,
    fps: Float?,
) {
    val metrics = buildString {
        append("推理耗时：")
        if (latencyMs == null) {
            append("-- ms")
        } else {
            append("${"%.1f".format(latencyMs)} ms")
        }
        fps?.let {
            append("    FPS：${"%.1f".format(it)}")
        }
    }
    Text(text = metrics, style = MaterialTheme.typography.bodyMedium)
}

@Composable
private fun DetectionPreview(
    bitmap: Bitmap?,
    detections: List<Detection>,
    modifier: Modifier = Modifier,
) {
    Box(modifier = modifier, contentAlignment = Alignment.Center) {
        if (bitmap == null) {
            Text("未选择图片或摄像头尚未出帧")
            return@Box
        }

        Image(
            bitmap = bitmap.asImageBitmap(),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Fit,
        )
        DetectionOverlay(
            bitmap = bitmap,
            detections = detections,
            modifier = Modifier.fillMaxSize(),
        )
    }
}

@Composable
private fun DetectionOverlay(
    bitmap: Bitmap,
    detections: List<Detection>,
    modifier: Modifier = Modifier,
) {
    Canvas(modifier = modifier) {
        val imageRatio = bitmap.width.toFloat() / bitmap.height.toFloat()
        val canvasRatio = size.width / size.height
        val colorClassCount = max(
            DefaultColorClassCount,
            detections.maxOfOrNull { it.classId + 1 } ?: DefaultColorClassCount,
        )
        val drawWidth: Float
        val drawHeight: Float
        if (canvasRatio > imageRatio) {
            drawHeight = size.height
            drawWidth = drawHeight * imageRatio
        } else {
            drawWidth = size.width
            drawHeight = drawWidth / imageRatio
        }
        val offsetX = (size.width - drawWidth) * 0.5f
        val offsetY = (size.height - drawHeight) * 0.5f
        val scaleX = drawWidth / bitmap.width
        val scaleY = drawHeight / bitmap.height
        val strokeWidth = 4.dp.toPx()

        detections.forEach { detection ->
            val left = offsetX + detection.left * scaleX
            val top = offsetY + detection.top * scaleY
            val right = offsetX + detection.right * scaleX
            val bottom = offsetY + detection.bottom * scaleY
            val detectionColor = colorForClass(detection.classId, colorClassCount)
            drawRect(
                color = detectionColor,
                topLeft = Offset(left, top),
                size = Size(width = right - left, height = bottom - top),
                style = Stroke(width = strokeWidth),
            )
            drawLabel(
                text = "${detection.label} ${"%.1f".format(detection.score * 100)}%",
                left = left,
                top = top,
                backgroundColor = detectionColor,
            )
        }
    }
}

private fun colorForClass(classId: Int, classCount: Int): Color {
    val safeClassCount = max(1, classCount)
    val hue = (classId.coerceAtLeast(0) % safeClassCount) * 360.0f / safeClassCount
    return Color.hsv(hue, 0.7f, 1.0f)
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawLabel(
    text: String,
    left: Float,
    top: Float,
    backgroundColor: Color,
) {
    val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = android.graphics.Color.BLACK
        textSize = 14.dp.toPx()
        style = Paint.Style.FILL
    }
    val backgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = backgroundColor.toAndroidColor(alpha = 220)
        style = Paint.Style.FILL
    }
    val padding = 4.dp.toPx()
    val baseline = (top - padding).coerceAtLeast(textPaint.textSize + padding)
    val textWidth = textPaint.measureText(text)
    drawIntoCanvas { canvas ->
        canvas.nativeCanvas.drawRect(
            left,
            baseline - textPaint.textSize - padding,
            left + textWidth + padding * 2,
            baseline + padding,
            backgroundPaint,
        )
        canvas.nativeCanvas.drawText(text, left + padding, baseline, textPaint)
    }
}

private fun Color.toAndroidColor(alpha: Int = 255): Int {
    return android.graphics.Color.argb(
        alpha,
        (red * 255).roundToInt().coerceIn(0, 255),
        (green * 255).roundToInt().coerceIn(0, 255),
        (blue * 255).roundToInt().coerceIn(0, 255),
    )
}

private fun ImageProxy.toBitmapArgb8888(): Bitmap {
    val plane = planes.first()
    val buffer: ByteBuffer = plane.buffer
    buffer.rewind()

    val bitmapWidth = if (plane.pixelStride > 0) plane.rowStride / plane.pixelStride else width
    val paddedBitmap = Bitmap.createBitmap(bitmapWidth, height, Bitmap.Config.ARGB_8888)
    paddedBitmap.copyPixelsFromBuffer(buffer)

    val croppedBitmap = if (bitmapWidth == width) {
        paddedBitmap
    } else {
        Bitmap.createBitmap(paddedBitmap, 0, 0, width, height)
    }
    val rotationDegrees = imageInfo.rotationDegrees
    if (rotationDegrees == 0) {
        return croppedBitmap
    }

    val matrix = Matrix().apply {
        postRotate(rotationDegrees.toFloat())
    }
    return Bitmap.createBitmap(croppedBitmap, 0, 0, croppedBitmap.width, croppedBitmap.height, matrix, true)
}

private fun decodeBitmap(context: Context, uri: Uri): Bitmap? {
    return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        val source = ImageDecoder.createSource(context.contentResolver, uri)
        ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            decoder.isMutableRequired = false
        }
    } else {
        @Suppress("DEPRECATION")
        MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
    } ?: context.contentResolver.openInputStream(uri)?.use(BitmapFactory::decodeStream)
}

@Preview(showBackground = true)
@Composable
fun YoloScreenPreview() {
    YoloappTheme {
        YoloScreen()
    }
}
