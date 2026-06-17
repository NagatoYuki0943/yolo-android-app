package com.example.yolo_app.detector

import android.content.Context
import android.graphics.Bitmap

class NcnnYoloDetector(
    context: Context,
    val useGpu: Boolean = true,
) : AutoCloseable {
    private var nativeHandle: Long = 0L

    init {
        nativeHandle = nativeCreate(
            context.assets,
            "yolo26n_ncnn_model/model.ncnn.param",
            "yolo26n_ncnn_model/model.ncnn.bin",
            "yolo26n_ncnn_model/metadata.yaml",
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
    }
}
