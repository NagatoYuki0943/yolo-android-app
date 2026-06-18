package com.example.yolo_app.detector

import android.content.Context
import android.graphics.Bitmap

enum class NcnnModelType(
    val displayName: String,
    val assetDir: String,
) {
    Float32("float32", "yolo26n_ncnn_model"),
    Int8("int8", "yolo26n_ncnn_model_int8"),
}

class NcnnYoloDetector(
    context: Context,
    val useGpu: Boolean = true,
    val modelType: NcnnModelType = NcnnModelType.Float32,
) : AutoCloseable {
    private var nativeHandle: Long = 0L
    val outputFormat: YoloOutputFormat = parseOutputFormat(context, modelType)

    init {
        nativeHandle = nativeCreate(
            context.assets,
            "${modelType.assetDir}/model.ncnn.param",
            "${modelType.assetDir}/model.ncnn.bin",
            "${modelType.assetDir}/metadata.yaml",
            useGpu,
        )
    }

    fun detect(
        bitmap: Bitmap,
        scoreThreshold: Float = 0.35f,
        nmsThreshold: Float = 0.45f,
    ): List<Detection> {
        check(nativeHandle != 0L) { "ncnn detector has already been closed" }
        return nativeDetect(nativeHandle, bitmap, scoreThreshold, nmsThreshold).toList()
    }

    override fun close() {
        val handle = nativeHandle
        if (handle != 0L) {
            nativeDestroy(handle)
            nativeHandle = 0L
        }
    }

    private external fun nativeCreate(
        assetManager: android.content.res.AssetManager,
        paramAssetPath: String,
        binAssetPath: String,
        metadataAssetPath: String,
        useVulkan: Boolean,
    ): Long

    private external fun nativeDetect(
        nativeHandle: Long,
        bitmap: Bitmap,
        scoreThreshold: Float,
        nmsThreshold: Float,
    ): Array<Detection>

    private external fun nativeDestroy(nativeHandle: Long)

    companion object {
        init {
            System.loadLibrary("yolo_ncnn")
        }

        private fun parseOutputFormat(context: Context, modelType: NcnnModelType): YoloOutputFormat {
            val metadataPath = "${modelType.assetDir}/metadata.yaml"
            val yaml = context.assets.open(metadataPath).bufferedReader().use { it.readText() }
            val endToEnd = yaml.lineSequence()
                .map { it.trim() }
                .firstOrNull { it.startsWith("end2end:") }
                ?.substringAfter(":")
                ?.trim()
                ?.equals("true", ignoreCase = true)
            return when (endToEnd) {
                true -> YoloOutputFormat.EndToEnd
                false -> YoloOutputFormat.Raw
                null -> YoloOutputFormat.Unknown
            }
        }
    }
}
