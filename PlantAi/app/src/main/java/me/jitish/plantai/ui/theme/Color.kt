package me.jitish.plantai.ui.theme

import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import android.graphics.Color as AndroidColor
import me.jitish.plantai.core.util.Constants

val DefaultSeedColor = Color(Constants.DEFAULT_SEED_COLOR_ARGB)

val ThemeSeedPresets = listOf(
    DefaultSeedColor,
    Color(0xFF2E7D32),
    Color(0xFF1565C0),
    Color(0xFF00897B),
    Color(0xFFC62828),
    Color(0xFFEF6C00),
    Color(0xFF6A1B9A),
    Color(0xFFAD1457),
    Color(0xFF455A64),
    Color(0xFF5D4037)
)

fun colorSchemeFromSeed(seed: Color, darkTheme: Boolean) =
    if (darkTheme) darkSchemeFromSeed(seed) else lightSchemeFromSeed(seed)

private fun lightSchemeFromSeed(seed: Color) = lightColorScheme(
    primary = seed.tone(40f),
    onPrimary = seed.tone(100f),
    primaryContainer = seed.tone(90f),
    onPrimaryContainer = seed.tone(10f),
    secondary = seed.shiftHue(20f).tone(40f),
    onSecondary = seed.shiftHue(20f).tone(100f),
    secondaryContainer = seed.shiftHue(20f).tone(90f),
    onSecondaryContainer = seed.shiftHue(20f).tone(10f),
    tertiary = seed.shiftHue(-40f).tone(40f),
    onTertiary = seed.shiftHue(-40f).tone(100f),
    tertiaryContainer = seed.shiftHue(-40f).tone(90f),
    onTertiaryContainer = seed.shiftHue(-40f).tone(10f)
)

private fun darkSchemeFromSeed(seed: Color) = darkColorScheme(
    primary = seed.tone(80f),
    onPrimary = seed.tone(20f),
    primaryContainer = seed.tone(30f),
    onPrimaryContainer = seed.tone(90f),
    secondary = seed.shiftHue(20f).tone(80f),
    onSecondary = seed.shiftHue(20f).tone(20f),
    secondaryContainer = seed.shiftHue(20f).tone(30f),
    onSecondaryContainer = seed.shiftHue(20f).tone(90f),
    tertiary = seed.shiftHue(-40f).tone(80f),
    onTertiary = seed.shiftHue(-40f).tone(20f),
    tertiaryContainer = seed.shiftHue(-40f).tone(30f),
    onTertiaryContainer = seed.shiftHue(-40f).tone(90f)
)

private fun Color.tone(lightnessPercent: Float): Color {
    val hsv = FloatArray(3)
    AndroidColor.colorToHSV(toArgb(), hsv)
    hsv[1] = (hsv[1] * when {
        lightnessPercent >= 90f -> 0.18f
        lightnessPercent >= 80f -> 0.35f
        lightnessPercent <= 20f -> 0.45f
        else -> 0.72f
    }).coerceIn(0f, 1f)
    hsv[2] = (lightnessPercent / 100f).coerceIn(0.08f, 1f)
    return Color(AndroidColor.HSVToColor(hsv))
}

private fun Color.shiftHue(degrees: Float): Color {
    val hsv = FloatArray(3)
    AndroidColor.colorToHSV(toArgb(), hsv)
    hsv[0] = (hsv[0] + degrees + 360f) % 360f
    return Color(AndroidColor.HSVToColor(hsv))
}

fun Color.hue(): Float {
    val hsv = FloatArray(3)
    AndroidColor.colorToHSV(toArgb(), hsv)
    return hsv[0]
}

fun colorFromHue(hue: Float): Color {
    val hsv = floatArrayOf(hue.coerceIn(0f, 360f), 0.72f, 0.72f)
    return Color(AndroidColor.HSVToColor(hsv))
}
