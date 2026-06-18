package com.example.yolo_app.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class OnnxYoloDetector(
    context: Context,
    val useNnapi: Boolean = true,
) : AutoCloseable {
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val modelFile: File = copyAssetToCache(context, MODEL_ASSET_PATH)
    private val sessionOptions = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setIntraOpNumThreads(4)
        setInterOpNumThreads(1)
        if (useNnapi) {
            addNnapi()
        }
    }
    private val session: OrtSession = environment.createSession(modelFile.absolutePath, sessionOptions)
    private val inputName: String = session.inputNames.first()
    private val outputName: String = session.outputNames.first()
    private val inputWidth: Int
    private val inputHeight: Int
    private val labels: List<String>
    private val endToEnd: Boolean
    var outputFormat: YoloOutputFormat = YoloOutputFormat.Unknown
        private set

    init {
        val metadata = session.metadata.customMetadata
        val imageSize = parseImageSize(metadata["imgsz"])
        inputWidth = imageSize.first
        inputHeight = imageSize.second
        labels = parseNames(metadata["names"])
        endToEnd = metadata["end2end"]?.equals("true", ignoreCase = true) == true
        require(labels.isNotEmpty()) { "Failed to read class names from ONNX metadata" }
    }

    fun detect(
        bitmap: Bitmap,
        scoreThreshold: Float = 0.35f,
        nmsThreshold: Float = 0.45f,
    ): List<Detection> {
        val preprocess = preprocess(bitmap)
        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(preprocess.input),
            longArrayOf(1L, 3L, inputHeight.toLong(), inputWidth.toLong()),
        ).use { inputTensor ->
            session.run(mapOf(inputName to inputTensor), setOf(outputName)).use { result ->
                val outputTensor = result[0] as OnnxTensor
                val shape = outputTensor.info.shape
                outputFormat = inferOutputFormat(shape, labels.size)
                val output = FloatArray(outputTensor.floatBuffer.remaining())
                outputTensor.floatBuffer.get(output)
                return parseOutput(
                    output = output,
                    shape = shape,
                    originalWidth = bitmap.width,
                    originalHeight = bitmap.height,
                    scale = preprocess.scale,
                    padX = preprocess.padX,
                    padY = preprocess.padY,
                    scoreThreshold = scoreThreshold,
                    nmsThreshold = nmsThreshold,
                )
            }
        }
    }

    override fun close() {
        session.close()
        sessionOptions.close()
    }

    private fun preprocess(bitmap: Bitmap): PreprocessResult {
        val scale = min(inputWidth.toFloat() / bitmap.width, inputHeight.toFloat() / bitmap.height)
        val resizedWidth = (bitmap.width * scale).roundToInt()
        val resizedHeight = (bitmap.height * scale).roundToInt()
        val padX = (inputWidth - resizedWidth) * 0.5f
        val padY = (inputHeight - resizedHeight) * 0.5f

        val inputBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(inputBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.drawBitmap(
            bitmap,
            null,
            RectF(padX, padY, padX + resizedWidth, padY + resizedHeight),
            Paint(Paint.FILTER_BITMAP_FLAG),
        )

        val pixels = IntArray(inputWidth * inputHeight)
        inputBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        val input = FloatArray(3 * inputWidth * inputHeight)
        val planeSize = inputWidth * inputHeight
        for (i in pixels.indices) {
            val pixel = pixels[i]
            input[i] = Color.red(pixel) / 255.0f
            input[planeSize + i] = Color.green(pixel) / 255.0f
            input[planeSize * 2 + i] = Color.blue(pixel) / 255.0f
        }
        inputBitmap.recycle()

        return PreprocessResult(
            input = input,
            scale = scale,
            padX = padX,
            padY = padY,
        )
    }

    private fun parseOutput(
        output: FloatArray,
        shape: LongArray,
        originalWidth: Int,
        originalHeight: Int,
        scale: Float,
        padX: Float,
        padY: Float,
        scoreThreshold: Float,
        nmsThreshold: Float,
    ): List<Detection> {
        val dims = shape.filter { it > 0L }.map { it.toInt() }
        val proposals = when {
            isRawYoloOutput(dims) -> parseRawOutput(
                output = output,
                shape = dims,
                originalWidth = originalWidth,
                originalHeight = originalHeight,
                scale = scale,
                padX = padX,
                padY = padY,
                scoreThreshold = scoreThreshold,
            )
            endToEnd || dims.takeLast(2).lastOrNull() == 6 -> parseEndToEndOutput(
                output = output,
                shape = dims,
                originalWidth = originalWidth,
                originalHeight = originalHeight,
                scale = scale,
                padX = padX,
                padY = padY,
                scoreThreshold = scoreThreshold,
            )
            else -> parseRawOutput(
                output = output,
                shape = dims,
                originalWidth = originalWidth,
                originalHeight = originalHeight,
                scale = scale,
                padX = padX,
                padY = padY,
                scoreThreshold = scoreThreshold,
            )
        }.sortedByDescending { it.score }

        return nonMaxSuppression(proposals, nmsThreshold)
    }

    private fun parseEndToEndOutput(
        output: FloatArray,
        shape: List<Int>,
        originalWidth: Int,
        originalHeight: Int,
        scale: Float,
        padX: Float,
        padY: Float,
        scoreThreshold: Float,
    ): List<Detection> {
        val rowCount = if (shape.size >= 2) shape[shape.size - 2] else output.size / 6
        val valuesPerRow = if (shape.isNotEmpty()) shape.last() else 6
        if (valuesPerRow < 6) return emptyList()

        return buildList {
            for (i in 0 until rowCount) {
                val base = i * valuesPerRow
                if (base + 5 >= output.size) break
                val score = output[base + 4]
                if (score < scoreThreshold) continue
                val classId = output[base + 5].roundToInt()
                add(
                    Detection(
                        label = labelFor(classId),
                        classId = classId,
                        score = score,
                        left = ((output[base] - padX) / scale).coerceIn(0.0f, originalWidth.toFloat()),
                        top = ((output[base + 1] - padY) / scale).coerceIn(0.0f, originalHeight.toFloat()),
                        right = ((output[base + 2] - padX) / scale).coerceIn(0.0f, originalWidth.toFloat()),
                        bottom = ((output[base + 3] - padY) / scale).coerceIn(0.0f, originalHeight.toFloat()),
                    ),
                )
            }
        }
    }

    private fun parseRawOutput(
        output: FloatArray,
        shape: List<Int>,
        originalWidth: Int,
        originalHeight: Int,
        scale: Float,
        padX: Float,
        padY: Float,
        scoreThreshold: Float,
    ): List<Detection> {
        val classCount = labels.size
        val outputChannels = 4 + classCount
        val lastTwo = shape.takeLast(2)
        val channelFirst = lastTwo.firstOrNull() == outputChannels
        val featureCount = if (channelFirst) lastTwo.getOrNull(1) ?: 0 else lastTwo.firstOrNull() ?: 0

        return buildList {
            for (i in 0 until featureCount) {
                val cx: Float
                val cy: Float
                val width: Float
                val height: Float
                var label = -1
                var score = 0.0f
                if (channelFirst) {
                    cx = output[i]
                    cy = output[featureCount + i]
                    width = output[featureCount * 2 + i]
                    height = output[featureCount * 3 + i]
                    for (classId in 0 until classCount) {
                        val classScore = output[featureCount * (4 + classId) + i]
                        if (classScore > score) {
                            score = classScore
                            label = classId
                        }
                    }
                } else {
                    val base = i * outputChannels
                    cx = output[base]
                    cy = output[base + 1]
                    width = output[base + 2]
                    height = output[base + 3]
                    for (classId in 0 until classCount) {
                        val classScore = output[base + 4 + classId]
                        if (classScore > score) {
                            score = classScore
                            label = classId
                        }
                    }
                }

                if (score < scoreThreshold) continue
                add(
                    Detection(
                        label = labelFor(label),
                        classId = label,
                        score = score,
                        left = ((cx - width * 0.5f - padX) / scale).coerceIn(0.0f, originalWidth.toFloat()),
                        top = ((cy - height * 0.5f - padY) / scale).coerceIn(0.0f, originalHeight.toFloat()),
                        right = ((cx + width * 0.5f - padX) / scale).coerceIn(0.0f, originalWidth.toFloat()),
                        bottom = ((cy + height * 0.5f - padY) / scale).coerceIn(0.0f, originalHeight.toFloat()),
                    ),
                )
            }
        }
    }

    private fun isRawYoloOutput(shape: List<Int>): Boolean {
        val outputChannels = 4 + labels.size
        val lastTwo = shape.takeLast(2)
        return lastTwo.firstOrNull() == outputChannels || lastTwo.getOrNull(1) == outputChannels
    }

    private fun labelFor(classId: Int): String {
        return labels.getOrNull(classId)?.takeIf { it.isNotBlank() } ?: "class_$classId"
    }

    private data class PreprocessResult(
        val input: FloatArray,
        val scale: Float,
        val padX: Float,
        val padY: Float,
    )

    companion object {
        private const val MODEL_ASSET_PATH = "yolo26n_onnx_model/model.onnx"

        private fun copyAssetToCache(context: Context, assetPath: String): File {
            val output = File(context.cacheDir, assetPath.substringAfterLast('/'))
            context.assets.open(assetPath).use { input ->
                output.outputStream().use { input.copyTo(it) }
            }
            return output
        }

        private fun parseImageSize(value: String?): Pair<Int, Int> {
            val numbers = value
                ?.let { Regex("\\d+").findAll(it).map { match -> match.value.toInt() }.toList() }
                .orEmpty()
            val width = numbers.getOrNull(0) ?: 640
            val height = numbers.getOrNull(1) ?: width
            return width to height
        }

        private fun parseNames(value: String?): List<String> {
            if (value.isNullOrBlank()) return emptyList()
            val namesById = sortedMapOf<Int, String>()
            val pattern = Regex("(\\d+)\\s*:\\s*['\"]([^'\"]+)['\"]")
            pattern.findAll(value).forEach { match ->
                namesById[match.groupValues[1].toInt()] = match.groupValues[2]
            }
            if (namesById.isEmpty()) return emptyList()
            val labels = MutableList((namesById.keys.maxOrNull() ?: -1) + 1) { "" }
            namesById.forEach { (id, name) -> labels[id] = name }
            return labels
        }

        private fun inferOutputFormat(shape: LongArray?, classCount: Int): YoloOutputFormat {
            val dims = shape?.filter { it > 0L }?.map { it.toInt() }.orEmpty()
            val lastTwo = dims.takeLast(2)
            val outputChannels = 4 + classCount
            return when {
                lastTwo.firstOrNull() == outputChannels || lastTwo.getOrNull(1) == outputChannels -> YoloOutputFormat.Raw
                lastTwo.lastOrNull() == 6 -> YoloOutputFormat.EndToEnd
                else -> YoloOutputFormat.Unknown
            }
        }

        private fun nonMaxSuppression(detections: List<Detection>, threshold: Float): List<Detection> {
            val picked = mutableListOf<Detection>()
            detections.forEach { candidate ->
                val keep = picked.none { selected -> iou(candidate, selected) > threshold }
                if (keep) {
                    picked += candidate
                }
            }
            return picked
        }

        private fun iou(a: Detection, b: Detection): Float {
            val left = max(a.left, b.left)
            val top = max(a.top, b.top)
            val right = min(a.right, b.right)
            val bottom = min(a.bottom, b.bottom)
            val intersection = max(0.0f, right - left) * max(0.0f, bottom - top)
            val areaA = max(0.0f, a.right - a.left) * max(0.0f, a.bottom - a.top)
            val areaB = max(0.0f, b.right - b.left) * max(0.0f, b.bottom - b.top)
            val union = areaA + areaB - intersection
            return if (union > 0.0f) intersection / union else 0.0f
        }
    }
}
