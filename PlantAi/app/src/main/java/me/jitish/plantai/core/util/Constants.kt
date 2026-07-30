package me.jitish.plantai.core.util

object Constants {
    const val DEFAULT_API_BASE_URL = "https://ai.miscweb.eu.org/"
    const val HEALTH_ENDPOINT = "/health"
    const val CLASSES_ENDPOINT = "/classes"
    const val PREDICT_ENDPOINT = "/predict"
    const val CONNECT_TIMEOUT_MS = 15_000
    const val READ_TIMEOUT_MS = 45_000

    const val PREFS_NAME = "plantai_settings"
    const val PREF_API_BASE_URL = "api_base_url"
    const val PREF_SEED_COLOR = "seed_color"

    /** Default Material seed color (#9D33B8). */
    const val DEFAULT_SEED_COLOR_ARGB = 0xFF9D33B8.toInt()
}
