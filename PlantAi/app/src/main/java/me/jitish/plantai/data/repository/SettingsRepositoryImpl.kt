package me.jitish.plantai.data.repository

import android.content.Context
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.core.content.edit
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import me.jitish.plantai.core.util.Constants
import me.jitish.plantai.domain.repository.SettingsRepository

class SettingsRepositoryImpl(context: Context) : SettingsRepository {
    private val prefs = context.getSharedPreferences(Constants.PREFS_NAME, Context.MODE_PRIVATE)

    private val _apiBaseUrl = MutableStateFlow(
        prefs.getString(Constants.PREF_API_BASE_URL, Constants.DEFAULT_API_BASE_URL)
            ?: Constants.DEFAULT_API_BASE_URL
    )
    private val _seedColor = MutableStateFlow(
        Color(prefs.getInt(Constants.PREF_SEED_COLOR, Constants.DEFAULT_SEED_COLOR_ARGB))
    )

    override val apiBaseUrl: StateFlow<String> = _apiBaseUrl.asStateFlow()
    override val seedColor: StateFlow<Color> = _seedColor.asStateFlow()

    override fun setApiBaseUrl(url: String) {
        prefs.edit { putString(Constants.PREF_API_BASE_URL, url) }
        _apiBaseUrl.value = url
    }

    override fun setSeedColor(color: Color) {
        prefs.edit { putInt(Constants.PREF_SEED_COLOR, color.toArgb()) }
        _seedColor.value = color
    }
}
