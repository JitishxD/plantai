package me.jitish.plantai.ui.navigation

import androidx.activity.compose.BackHandler
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import me.jitish.plantai.MainActivity
import me.jitish.plantai.data.repository.PlantRepositoryImpl
import me.jitish.plantai.data.repository.SettingsRepositoryImpl
import me.jitish.plantai.ui.screens.home.HomeScreen
import me.jitish.plantai.ui.screens.settings.SettingsScreen
import me.jitish.plantai.ui.theme.PlantAiTheme

private enum class AppScreen { Home, Settings }

@Composable
fun PlantAiApp(activity: MainActivity) {
    val plantRepository = remember { PlantRepositoryImpl(activity) }
    val settingsRepository = remember { SettingsRepositoryImpl(activity) }
    val apiBaseUrl by settingsRepository.apiBaseUrl.collectAsState()
    val seedColor by settingsRepository.seedColor.collectAsState()
    var screen by remember { mutableStateOf(AppScreen.Home) }

    PlantAiTheme(seedColor = seedColor) {
        BackHandler(enabled = screen == AppScreen.Settings) {
            screen = AppScreen.Home
        }

        when (screen) {
            AppScreen.Home -> HomeScreen(
                activity = activity,
                plantRepository = plantRepository,
                apiBaseUrl = apiBaseUrl,
                onOpenSettings = { screen = AppScreen.Settings }
            )
            AppScreen.Settings -> SettingsScreen(
                apiBaseUrl = apiBaseUrl,
                seedColor = seedColor,
                onApiBaseUrlChange = settingsRepository::setApiBaseUrl,
                onSeedColorChange = settingsRepository::setSeedColor,
                onBack = { screen = AppScreen.Home }
            )
        }
    }
}
