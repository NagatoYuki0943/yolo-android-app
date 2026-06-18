package com.example.yolo_app.detector

enum class YoloOutputFormat(
    val displayName: String,
) {
    Raw("raw"),
    EndToEnd("end2end"),
    Unknown("auto"),
}
