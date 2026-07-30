package me.jitish.plantai.domain.repository

import androidx.compose.ui.graphics.Color
import kotlinx.coroutines.flow.StateFlow

interface SettingsRepository {
    val apiBaseUrl: StateFlow<String>
    val seedColor: StateFlow<Color>

    fun setApiBaseUrl(url: String)
    fun setSeedColor(color: Color)
}
