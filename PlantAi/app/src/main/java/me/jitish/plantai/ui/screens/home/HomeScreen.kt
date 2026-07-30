package me.jitish.plantai.ui.screens.home

import android.content.ClipData
import android.content.ClipboardManager
import android.net.Uri
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.clickable
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogProperties
import androidx.core.content.FileProvider
import java.io.File
import me.jitish.plantai.MainActivity
import me.jitish.plantai.data.repository.toDisplayName
import me.jitish.plantai.domain.model.ClassCatalog
import me.jitish.plantai.domain.model.LeafDiagnosis
import me.jitish.plantai.domain.model.ServerHealth
import me.jitish.plantai.domain.repository.PlantRepository

@Composable
fun HomeScreen(
    activity: MainActivity,
    plantRepository: PlantRepository,
    apiBaseUrl: String,
    onOpenSettings: () -> Unit
) {
    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var cameraUri by remember { mutableStateOf<Uri?>(null) }
    var showInfo by remember { mutableStateOf(false) }
    var diagnosis by remember { mutableStateOf<LeafDiagnosis?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var loading by remember { mutableStateOf(false) }
    var health by remember { mutableStateOf<ServerHealth?>(null) }
    var catalog by remember { mutableStateOf<ClassCatalog?>(null) }
    var infoPanel by remember { mutableStateOf(InfoPanel.None) }
    var infoBusy by remember { mutableStateOf(false) }
    var infoError by remember { mutableStateOf<String?>(null) }

    val gallery = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) { imageUri = uri; diagnosis = null; error = null }
    }
    val camera = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) { imageUri = cameraUri; diagnosis = null; error = null }
    }
    val context = LocalContext.current
    val previewBitmap = remember(imageUri) { imageUri?.let { loadPreviewBitmap(context, it) } }

    fun runInfoApi(block: () -> Unit) {
        infoBusy = true
        infoError = null
        Thread {
            runCatching(block)
                .onFailure { throwable ->
                    activity.runOnUiThread {
                        infoBusy = false
                        infoError = throwable.message ?: "Request failed."
                    }
                }
        }.start()
    }

    if (showInfo) {
        ServerInfoDialog(
            panel = infoPanel,
            health = health,
            catalog = catalog,
            busy = infoBusy,
            error = infoError,
            onCheckHealth = {
                infoPanel = InfoPanel.Health
                catalog = null
                runInfoApi {
                    val result = plantRepository.health(apiBaseUrl)
                    activity.runOnUiThread { health = result; infoBusy = false }
                }
            },
            onLoadClasses = {
                infoPanel = InfoPanel.Classes
                health = null
                runInfoApi {
                    val result = plantRepository.classes(apiBaseUrl)
                    activity.runOnUiThread { catalog = result; infoBusy = false }
                }
            },
            onDismiss = { showInfo = false }
        )
    }

    Scaffold(topBar = {
        Row(
            Modifier.fillMaxWidth().statusBarsPadding().padding(horizontal = 20.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(Icons.Default.Eco, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(30.dp))
            Spacer(Modifier.width(10.dp))
            Text("PlantAI", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
            Spacer(Modifier.weight(1f))
            IconButton(onClick = { showInfo = true }) { Icon(Icons.Default.Info, "Server info") }
            IconButton(onClick = onOpenSettings) { Icon(Icons.Default.Settings, "Settings") }
        }
    }) { padding ->
        Column(
            Modifier.fillMaxSize().padding(padding).padding(horizontal = 20.dp).verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                "Your plant health check",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.fillMaxWidth().padding(top = 16.dp)
            )
            Text(
                "Take a clear photo of one affected leaf in natural light.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.fillMaxWidth().padding(top = 6.dp)
            )
            Spacer(Modifier.height(24.dp))

            Card(Modifier.fillMaxWidth().height(210.dp)) {
                Box(
                    Modifier.fillMaxSize().clickable {
                        gallery.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                    },
                    contentAlignment = Alignment.Center
                ) {
                    if (previewBitmap != null) {
                        Image(
                            previewBitmap.asImageBitmap(),
                            contentDescription = "Selected plant leaf",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
                        )
                    } else {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Icon(Icons.Default.Image, null, modifier = Modifier.size(54.dp), tint = MaterialTheme.colorScheme.primary)
                            Text("Choose a leaf image", fontWeight = FontWeight.Medium)
                            Text("Tap to choose from your gallery", color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }

            Spacer(Modifier.height(14.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedButton(
                    onClick = { gallery.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Default.Image, null); Spacer(Modifier.width(8.dp)); Text("Gallery")
                }
                OutlinedButton(
                    onClick = {
                        val file = File(activity.cacheDir, "camera/leaf_${System.currentTimeMillis()}.jpg").also { it.parentFile?.mkdirs() }
                        cameraUri = FileProvider.getUriForFile(activity, "${activity.packageName}.fileprovider", file)
                        camera.launch(cameraUri)
                    },
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Default.CameraAlt, null); Spacer(Modifier.width(8.dp)); Text("Camera")
                }
            }

            Spacer(Modifier.height(14.dp))
            Button(
                onClick = {
                    val uri = imageUri ?: run { error = "Choose or take a leaf photo first."; return@Button }
                    loading = true
                    error = null
                    diagnosis = null
                    Thread {
                        runCatching { plantRepository.predict(uri, apiBaseUrl) }
                            .onSuccess { result -> activity.runOnUiThread { diagnosis = result; loading = false } }
                            .onFailure { throwable ->
                                activity.runOnUiThread {
                                    loading = false
                                    error = throwable.message ?: "Could not reach the diagnosis service."
                                }
                            }
                    }.start()
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = !loading
            ) {
                if (loading) CircularProgressIndicator(Modifier.size(22.dp), strokeWidth = 2.dp)
                else Text("Analyze leaf")
            }

            error?.let { CopyableError(it, modifier = Modifier.padding(top = 16.dp), centered = true) }
            diagnosis?.let { ResultCard(it) }
            Spacer(Modifier.height(28.dp))
        }
    }
}

private enum class InfoPanel { None, Health, Classes }

@Composable
private fun ServerInfoDialog(
    panel: InfoPanel,
    health: ServerHealth?,
    catalog: ClassCatalog?,
    busy: Boolean,
    error: String?,
    onCheckHealth: () -> Unit,
    onLoadClasses: () -> Unit,
    onDismiss: () -> Unit
) {
    val scrollState = rememberScrollState()
    AlertDialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false),
        modifier = Modifier.fillMaxWidth(0.92f),
        icon = { Icon(Icons.Default.Info, contentDescription = null) },
        title = { Text("Server info") },
        text = {
            Column {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    OutlinedButton(onClick = onCheckHealth, enabled = !busy, modifier = Modifier.weight(1f)) {
                        Text("Health")
                    }
                    OutlinedButton(onClick = onLoadClasses, enabled = !busy, modifier = Modifier.weight(1f)) {
                        Text("Classes")
                    }
                }
                if (busy) {
                    LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 12.dp))
                }
                error?.let { CopyableError(it, modifier = Modifier.padding(top = 12.dp)) }
                Box(
                    Modifier
                        .fillMaxWidth()
                        .heightIn(max = 320.dp)
                        .padding(top = 12.dp)
                        .verticalScrollbar(scrollState)
                ) {
                    SelectionContainer {
                        Column(Modifier.verticalScroll(scrollState).padding(end = 10.dp)) {
                            if (panel == InfoPanel.Health) {
                                health?.let { status ->
                                    Text(
                                        if (status.ready) "Server ready" else "Server degraded",
                                        fontWeight = FontWeight.Bold,
                                        color = if (status.ready) {
                                            MaterialTheme.colorScheme.primary
                                        } else {
                                            MaterialTheme.colorScheme.error
                                        }
                                    )
                                    HealthLine("Status", status.status)
                                    HealthLine("Version", status.version)
                                    HealthLine("Ready", status.ready.toString())
                                    HealthLine("Classes loaded", status.classesLoaded.toString())
                                    HealthLine("Backbone", status.backbone)
                                    HealthLine("Image size", "${status.imgSize}px")
                                    HealthLine("TTA", if (status.useTta) "on" else "off")
                                    HealthLine(
                                        "Confidence threshold",
                                        "${(status.confidenceThreshold * 100).toInt()}%"
                                    )
                                }
                            }
                            if (panel == InfoPanel.Classes) {
                                catalog?.let { list ->
                                    Text("${list.count} disease classes", fontWeight = FontWeight.Bold)
                                    list.classes.forEach { label ->
                                        Text(
                                            label.toDisplayName(),
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            modifier = Modifier.padding(top = 6.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Close") }
        }
    )
}

@Composable
private fun CopyableError(
    message: String,
    modifier: Modifier = Modifier,
    centered: Boolean = false
) {
    val context = LocalContext.current
    Row(
        modifier = modifier.fillMaxWidth(),
        verticalAlignment = Alignment.Top,
        horizontalArrangement = if (centered) Arrangement.Center else Arrangement.Start
    ) {
        SelectionContainer(Modifier.weight(1f, fill = false)) {
            Text(
                message,
                color = MaterialTheme.colorScheme.error,
                textAlign = if (centered) TextAlign.Center else TextAlign.Start
            )
        }
        IconButton(
            onClick = {
                val clipboard = context.getSystemService(ClipboardManager::class.java)
                clipboard.setPrimaryClip(ClipData.newPlainText("error", message))
            },
            modifier = Modifier.size(36.dp)
        ) {
            Icon(
                Icons.Default.ContentCopy,
                contentDescription = "Copy error",
                tint = MaterialTheme.colorScheme.error,
                modifier = Modifier.size(18.dp)
            )
        }
    }
}

@Composable
private fun HealthLine(label: String, value: String) {
    Text(
        "$label: $value",
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(top = 6.dp)
    )
}

@Composable
private fun Modifier.verticalScrollbar(
    state: ScrollState,
    width: Dp = 3.dp,
    color: Color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.45f)
): Modifier {
    val widthPx = with(LocalDensity.current) { width.toPx() }
    return drawWithContent {
        drawContent()
        val viewport = size.height
        val content = viewport + state.maxValue
        if (content <= viewport || state.maxValue <= 0) return@drawWithContent
        val thumbHeight = (viewport * viewport / content).coerceAtLeast(24f)
        val travel = (viewport - thumbHeight).coerceAtLeast(0f)
        val thumbOffset = travel * (state.value.toFloat() / state.maxValue.toFloat())
        drawRoundRect(
            color = color,
            topLeft = Offset(size.width - widthPx, thumbOffset),
            size = Size(widthPx, thumbHeight),
            cornerRadius = CornerRadius(widthPx / 2f, widthPx / 2f)
        )
    }
}

@Composable
private fun ResultCard(result: LeafDiagnosis) {
    Card(Modifier.fillMaxWidth().padding(top = 22.dp)) {
        Column(Modifier.padding(20.dp)) {
            Text(
                if (result.lowConfidence) "Possible diagnosis" else "Diagnosis",
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold
            )
            Text(
                result.disease.toDisplayName(),
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(top = 6.dp)
            )
            Text(
                "${(result.confidence * 100).toInt()}% confidence",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp)
            )
            if (result.lowConfidence) {
                Text(
                    "The photo is uncertain. Try another clear, close-up leaf photo.",
                    color = MaterialTheme.colorScheme.tertiary,
                    modifier = Modifier.padding(top = 12.dp)
                )
            }
            Text("Care guidance", fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 18.dp))
            Text(result.remedy, modifier = Modifier.padding(top = 4.dp))
            if (result.alternatives.isNotEmpty()) {
                Text(
                    "Also considered: ${result.alternatives.joinToString()}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 14.dp)
                )
            }
        }
    }
}

private fun loadPreviewBitmap(context: android.content.Context, uri: Uri): Bitmap? = runCatching {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
    val largestSide = maxOf(bounds.outWidth, bounds.outHeight)
    val sampleSize = generateSequence(1) { it * 2 }.takeWhile { it * 2 <= largestSide / 768 }.lastOrNull() ?: 1
    context.contentResolver.openInputStream(uri)?.use {
        BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = sampleSize })
    }
}.getOrNull()
