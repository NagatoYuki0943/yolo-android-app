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
import com.example.yolo_app.detector.NcnnYoloDetector
import com.example.yolo_app.ui.theme.YoloappTheme
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

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

private data class TimedDetectionResult(
    val detections: List<Detection>,
    val elapsedMs: Float,
    val device: InferenceDevice,
    val fallbackMessage: String? = null,
)

private data class InferenceOptions(
    val device: InferenceDevice,
    val scoreThreshold: Float,
    val iouThreshold: Float,
)

private class DetectorHolder(
    private val context: Context,
) : AutoCloseable {
    private var detector: NcnnYoloDetector? = null
    private var device: InferenceDevice? = null

    @Synchronized
    fun closeCurrent() {
        detector?.close()
        detector = null
        device = null
    }

    @Synchronized
    fun detect(
        bitmap: Bitmap,
        requestedDevice: InferenceDevice,
        scoreThreshold: Float,
        iouThreshold: Float,
    ): TimedDetectionResult {
        return tryDetect(
            bitmap = bitmap,
            requestedDevice = requestedDevice,
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
        requestedDevice: InferenceDevice,
        scoreThreshold: Float,
        iouThreshold: Float,
        fallbackMessage: String?,
    ): TimedDetectionResult {
        return try {
            val activeDetector = ensureDetector(requestedDevice)
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
                device = requestedDevice,
                fallbackMessage = fallbackMessage,
            )
        } catch (error: Throwable) {
            closeCurrent()
            if (requestedDevice == InferenceDevice.Gpu) {
                tryDetect(
                    bitmap = bitmap,
                    requestedDevice = InferenceDevice.Cpu,
                    scoreThreshold = scoreThreshold,
                    iouThreshold = iouThreshold,
                    fallbackMessage = "GPU 推理失败，已切换到 CPU：${error.message ?: error.javaClass.simpleName}",
                )
            } else {
                throw error
            }
        }
    }

    private fun ensureDetector(requestedDevice: InferenceDevice): NcnnYoloDetector {
        val activeDetector = detector
        if (activeDetector != null && device == requestedDevice) {
            return activeDetector
        }

        closeCurrent()
        return NcnnYoloDetector(context, useGpu = requestedDevice.useGpu).also {
            detector = it
            device = requestedDevice
        }
    }
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
    var inferenceDevice by remember { mutableStateOf(InferenceDevice.Gpu) }
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
        cameraRunning = granted
        status = if (granted) "摄像头权限已授予，正在启动实时推理" else "需要摄像头权限才能实时推理"
    }

    DisposableEffect(Unit) {
        onDispose {
            detectorHolder.close()
        }
    }

    if (mode == InferenceMode.Camera && hasCameraPermission) {
        CameraInferenceEffect(
            context = context,
            detectorProvider = {
                detectorHolder
            },
            enabled = cameraRunning,
            options = InferenceOptions(
                device = inferenceDevice,
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
                status = result.fallbackMessage ?: "实时推理中：${result.detections.size} 个目标，${inferenceDevice.displayName}"
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
        Text(text = "YOLO ncnn Demo", style = MaterialTheme.typography.headlineSmall)
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
                        cameraRunning = true
                        status = "正在启动摄像头"
                    }
                },
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
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            InferenceDevice.entries.forEach { device ->
                ModeButton(
                    text = device.displayName,
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
                                "将从下一帧切换到 ${device.displayName} 推理"
                            } else {
                                "已切换到 ${device.displayName} 推理"
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

        if (mode == InferenceMode.Image) {
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
                                    requestedDevice = inferenceDevice,
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
                                    ?: "图片检测完成：${result.detections.size} 个目标，${result.device.displayName}，${"%.1f".format(result.elapsedMs)} ms"
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
        DetectionPreview(
            bitmap = previewBitmap,
            detections = detections,
            latencyMs = latencyMs,
            fps = fps,
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
        val cameraExecutor = Executors.newSingleThreadExecutor()
        val mainHandler = Handler(Looper.getMainLooper())
        val isActive = AtomicBoolean(true)
        val isDetecting = AtomicBoolean(false)
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
                                requestedDevice = currentOptions.device,
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
            if (cameraProviderFuture.isDone) {
                cameraProviderFuture.get().unbindAll()
            }
            cameraExecutor.shutdown()
        }
    }
}

@Composable
private fun DetectionPreview(
    bitmap: Bitmap?,
    detections: List<Detection>,
    latencyMs: Float?,
    fps: Float?,
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
            latencyMs = latencyMs,
            fps = fps,
            modifier = Modifier.fillMaxSize(),
        )
    }
}

@Composable
private fun DetectionOverlay(
    bitmap: Bitmap,
    detections: List<Detection>,
    latencyMs: Float?,
    fps: Float?,
    modifier: Modifier = Modifier,
) {
    Canvas(modifier = modifier) {
        val imageRatio = bitmap.width.toFloat() / bitmap.height.toFloat()
        val canvasRatio = size.width / size.height
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
            drawRect(
                color = Color(0xFFFFD54F),
                topLeft = Offset(left, top),
                size = Size(width = right - left, height = bottom - top),
                style = Stroke(width = strokeWidth),
            )
            drawLabel(
                text = "${detection.label} ${"%.1f".format(detection.score * 100)}%",
                left = left,
                top = top,
            )
        }

        val metrics = buildString {
            latencyMs?.let { append("Infer ${"%.1f".format(it)} ms") }
            fps?.let {
                if (isNotEmpty()) append("  ")
                append("FPS ${"%.1f".format(it)}")
            }
        }
        if (metrics.isNotEmpty()) {
            drawLabel(text = metrics, left = offsetX + 8.dp.toPx(), top = offsetY + 8.dp.toPx())
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawLabel(
    text: String,
    left: Float,
    top: Float,
) {
    val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = android.graphics.Color.BLACK
        textSize = 14.dp.toPx()
        style = Paint.Style.FILL
    }
    val backgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = android.graphics.Color.argb(220, 255, 213, 79)
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
