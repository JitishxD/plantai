package me.jitish.plantai.data.repository

import android.content.Context
import android.net.Uri
import java.io.DataOutputStream
import java.net.HttpURLConnection
import java.net.URL
import me.jitish.plantai.core.util.Constants
import me.jitish.plantai.domain.model.ClassCatalog
import me.jitish.plantai.domain.model.LeafDiagnosis
import me.jitish.plantai.domain.model.ServerHealth
import me.jitish.plantai.domain.repository.PlantRepository
import org.json.JSONObject

class PlantRepositoryImpl(private val context: Context) : PlantRepository {
    override fun health(baseUrl: String): ServerHealth {
        val json = getJson(baseUrl, Constants.HEALTH_ENDPOINT)
        return ServerHealth(
            status = json.getString("status"),
            version = json.optString("version", "unknown"),
            ready = json.optBoolean("ready"),
            classesLoaded = json.optInt("classes_loaded", json.optInt("classes")),
            backbone = json.optString("backbone", "unknown"),
            imgSize = json.optInt("img_size"),
            useTta = json.optBoolean("use_tta"),
            confidenceThreshold = json.optDouble("confidence_threshold", 0.6)
        )
    }

    override fun classes(baseUrl: String): ClassCatalog {
        val json = getJson(baseUrl, Constants.CLASSES_ENDPOINT)
        val labels = json.optJSONArray("classes")
        val classes = buildList {
            if (labels != null) for (index in 0 until labels.length()) {
                add(labels.getString(index))
            }
        }
        return ClassCatalog(count = json.optInt("count", classes.size), classes = classes)
    }

    override fun predict(imageUri: Uri, baseUrl: String): LeafDiagnosis {
        val boundary = "PlantAi-${System.currentTimeMillis()}"
        val connection = (URL(baseUrl.trimEnd('/') + Constants.PREDICT_ENDPOINT).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            connectTimeout = Constants.CONNECT_TIMEOUT_MS
            readTimeout = Constants.READ_TIMEOUT_MS
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
        }
        DataOutputStream(connection.outputStream).use { output ->
            output.writeBytes("--$boundary\r\nContent-Disposition: form-data; name=\"image\"; filename=\"leaf.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n")
            context.contentResolver.openInputStream(imageUri)?.use { it.copyTo(output) }
                ?: throw IllegalArgumentException("Could not read the selected image.")
            output.writeBytes("\r\n--$boundary--\r\n")
        }
        val json = readJsonResponse(connection)
        val topK = json.optJSONArray("top_k")
        val alternatives = buildList {
            if (topK != null) for (index in 1 until topK.length()) {
                add(topK.getJSONObject(index).optString("disease").toDisplayName())
            }
        }
        return LeafDiagnosis(
            disease = json.getString("disease"),
            confidence = json.getDouble("confidence"),
            remedy = json.optString("remedy", "No care guidance available."),
            lowConfidence = json.optBoolean("low_confidence"),
            alternatives = alternatives
        )
    }

    private fun getJson(baseUrl: String, endpoint: String): JSONObject {
        val connection = (URL(baseUrl.trimEnd('/') + endpoint).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = Constants.CONNECT_TIMEOUT_MS
            readTimeout = Constants.READ_TIMEOUT_MS
        }
        return readJsonResponse(connection)
    }

    private fun readJsonResponse(connection: HttpURLConnection): JSONObject {
        val status = connection.responseCode
        val body = (if (status in 200..299) connection.inputStream else connection.errorStream)
            ?.bufferedReader()?.use { it.readText() }.orEmpty()
        if (status !in 200..299) {
            val message = runCatching {
                JSONObject(body).optString("error").ifBlank { JSONObject(body).optString("status") }
            }.getOrDefault("").ifBlank { "Server returned HTTP $status" }
            throw IllegalStateException(message)
        }
        return JSONObject(body)
    }
}

fun String.toDisplayName(): String = replace("___", " — ").replace("_", " ")
