package me.jitish.plantai.domain.model

data class LeafDiagnosis(
    val disease: String,
    val confidence: Double,
    val remedy: String,
    val lowConfidence: Boolean,
    val alternatives: List<String>
)
