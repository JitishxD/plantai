package me.jitish.plantai.domain.model

data class ServerHealth(
    val status: String,
    val version: String,
    val ready: Boolean,
    val classesLoaded: Int,
    val backbone: String,
    val imgSize: Int,
    val useTta: Boolean,
    val confidenceThreshold: Double
)
