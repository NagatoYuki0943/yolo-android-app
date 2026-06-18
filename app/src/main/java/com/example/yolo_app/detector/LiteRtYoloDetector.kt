package com.example.yolo_app.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

enum class LiteRtModelType(
    val displayName: String,
    val assetDir: String,
) {
    Float32("float32", "yolo26n_tflite_model"),
    Int8("int8", "yolo26n_tflite_model_int8"),
}

class LiteRtYoloDetector(
    context: Context,
    val useNnapi: Boolean = true,
    val modelType: LiteRtModelType = LiteRtModelType.Float32,
) : AutoCloseable {
    private val interpreter = Interpreter(
        loadModel(context, modelType),
        Interpreter.Options().apply {
            setNumThreads(4)
            setUseNNAPI(useNnapi)
        },
    )
    private val inputTensor: Tensor = interpreter.getInputTensor(0)
    private val outputTensor: Tensor = interpreter.getOutputTensor(0)
    private val inputShape = inputTensor.shape()
    private val outputShape = outputTensor.shape()
    private val inputDataType = inputTensor.dataType()
    private val outputDataType = outputTensor.dataType()
    private val inputQuantization = inputTensor.quantizationParams()
    private val outputQuantization = outputTensor.quantizationParams()
    private val inputWidth: Int
    private val inputHeight: Int
    private val labels: List<String>
    val outputFormat: YoloOutputFormat

    init {
        val metadata = parseMetadata(context, modelType)
        val imageSize = parseImageSize(metadata["imgsz"])
        labels = parseNames(metadata["names"])
        inputHeight = inputShape.getOrNull(1) ?: imageSize.second
        inputWidth = inputShape.getOrNull(2) ?: imageSize.first
        require(labels.isNotEmpty()) { "Failed to read class names from LiteRT metadata.yaml" }
        outputFormat = inferOutputFormat(outputShape, labels.size)
    }

    fun detect(
        bitmap: Bitmap,
        scoreThreshold: Float = 0.35f,
        nmsThreshold: Float = 0.45f,
    ): List<Detection> {
        val preprocess = preprocess(bitmap)
        val inputBuffer = createInputBuffer(preprocess.input)
        val outputBuffer = ByteBuffer
            .allocateDirect(outputElementCount() * outputDataType.byteSize())
            .order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        return parseOutput(
            output = readOutput(outputBuffer),
            shape = outputShape,
            originalWidth = bitmap.width,
            originalHeight = bitmap.height,
            scale = preprocess.scale,
            padX = preprocess.padX,
            padY = preprocess.padY,
            scoreThreshold = scoreThreshold,
            nmsThreshold = nmsThreshold,
        )
    }

    override fun close() {
        interpreter.close()
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
        var outputIndex = 0
        for (pixel in pixels) {
            input[outputIndex++] = Color.red(pixel) / 255.0f
            input[outputIndex++] = Color.green(pixel) / 255.0f
            input[outputIndex++] = Color.blue(pixel) / 255.0f
        }
        inputBitmap.recycle()

        return PreprocessResult(
            input = input,
            scale = scale,
            padX = padX,
            padY = padY,
        )
    }

    private fun createInputBuffer(input: FloatArray): ByteBuffer {
        val buffer = ByteBuffer
            .allocateDirect(input.size * inputDataType.byteSize())
            .order(ByteOrder.nativeOrder())
        when (inputDataType) {
            DataType.FLOAT32 -> input.forEach { buffer.putFloat(it) }
            DataType.UINT8 -> input.forEach { value ->
                buffer.put(quantize(value, inputQuantization, 0, 255).toByte())
            }
            DataType.INT8 -> input.forEach { value ->
                buffer.put(quantize(value, inputQuantization, -128, 127).toByte())
            }
            else -> error("Unsupported LiteRT input type: $inputDataType")
        }
        buffer.rewind()
        return buffer
    }

    private fun readOutput(buffer: ByteBuffer): FloatArray {
        return when (outputDataType) {
            DataType.FLOAT32 -> FloatArray(outputElementCount()).also {
                buffer.asFloatBuffer().get(it)
            }
            DataType.UINT8 -> FloatArray(outputElementCount()) {
                dequantize(buffer.get().toInt() and 0xFF, outputQuantization)
            }
            DataType.INT8 -> FloatArray(outputElementCount()) {
                dequantize(buffer.get().toInt(), outputQuantization)
            }
            else -> error("Unsupported LiteRT output type: $outputDataType")
        }
    }

    private fun parseOutput(
        output: FloatArray,
        shape: IntArray,
        originalWidth: Int,
        originalHeight: Int,
        scale: Float,
        padX: Float,
        padY: Float,
        scoreThreshold: Float,
        nmsThreshold: Float,
    ): List<Detection> {
        val dims = shape.filter { it > 0 }
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
            dims.takeLast(2).lastOrNull() == 6 -> parseEndToEndOutput(
                output = output,
                shape = dims,
                originalWidth = originalWidth,
                originalHeight = originalHeight,
                scale = scale,
                padX = padX,
                padY = padY,
                scoreThreshold = scoreThreshold,
            )
            else -> emptyList()
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

    private fun outputElementCount(): Int {
        return outputShape.fold(1) { acc, value -> acc * value }
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
        private const val MODEL_FILE_NAME = "model.tflite"
        private const val METADATA_FILE_NAME = "metadata.yaml"

        private fun loadModel(context: Context, modelType: LiteRtModelType): MappedByteBuffer {
            val descriptor = context.assets.openFd("${modelType.assetDir}/$MODEL_FILE_NAME")
            FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
                return channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
            }
        }

        private fun parseMetadata(context: Context, modelType: LiteRtModelType): Map<String, String> {
            val yaml = context.assets.open("${modelType.assetDir}/$METADATA_FILE_NAME").bufferedReader().use { it.readText() }
            val values = mutableMapOf<String, String>()
            val names = sortedMapOf<Int, String>()
            val imageSizeValues = mutableListOf<Int>()
            var inNames = false
            var inImageSize = false
            yaml.lineSequence().forEach { line ->
                val trimmed = line.trim()
                when {
                    trimmed == "names:" -> {
                        inNames = true
                        inImageSize = false
                    }
                    trimmed == "imgsz:" -> {
                        inImageSize = true
                        inNames = false
                    }
                    !line.startsWith(" ") && trimmed.endsWith(":").not() -> {
                        inNames = false
                        inImageSize = false
                    }
                    inImageSize && trimmed.startsWith("- ") -> {
                        trimmed.removePrefix("- ").toIntOrNull()?.let { imageSizeValues += it }
                    }
                    inNames && trimmed.contains(":") -> {
                        val id = trimmed.substringBefore(":").toIntOrNull()
                        val name = trimmed.substringAfter(":").trim().trim('\'', '"')
                        if (id != null && name.isNotBlank()) {
                            names[id] = name
                        }
                    }
                }
            }
            if (imageSizeValues.isNotEmpty()) {
                values["imgsz"] = imageSizeValues.joinToString(prefix = "[", postfix = "]")
            }
            if (names.isNotEmpty()) {
                values["names"] = names.entries.joinToString(prefix = "{", postfix = "}") { (id, name) ->
                    "$id: '$name'"
                }
            }
            return values
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

        private fun inferOutputFormat(shape: IntArray, classCount: Int): YoloOutputFormat {
            val dims = shape.filter { it > 0 }
            val lastTwo = dims.takeLast(2)
            val outputChannels = 4 + classCount
            return when {
                lastTwo.firstOrNull() == outputChannels || lastTwo.getOrNull(1) == outputChannels -> YoloOutputFormat.Raw
                lastTwo.lastOrNull() == 6 -> YoloOutputFormat.EndToEnd
                else -> YoloOutputFormat.Unknown
            }
        }

        private fun quantize(
            value: Float,
            quantization: Tensor.QuantizationParams,
            minValue: Int,
            maxValue: Int,
        ): Int {
            val scale = quantization.scale
            val zeroPoint = quantization.zeroPoint
            if (scale == 0.0f) {
                return (value * 255.0f).roundToInt().coerceIn(minValue, maxValue)
            }
            return (value / scale + zeroPoint).roundToInt().coerceIn(minValue, maxValue)
        }

        private fun dequantize(value: Int, quantization: Tensor.QuantizationParams): Float {
            val scale = quantization.scale
            val zeroPoint = quantization.zeroPoint
            return if (scale == 0.0f) value.toFloat() else (value - zeroPoint) * scale
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
