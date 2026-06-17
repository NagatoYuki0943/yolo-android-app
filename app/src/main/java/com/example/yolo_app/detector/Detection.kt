package com.example.yolo_app.detector

data class Detection(
    val label: String,
    val classId: Int,
    val score: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
)
