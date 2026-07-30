package me.jitish.plantai.domain.repository

import android.net.Uri
import me.jitish.plantai.domain.model.ClassCatalog
import me.jitish.plantai.domain.model.LeafDiagnosis
import me.jitish.plantai.domain.model.ServerHealth

interface PlantRepository {
    fun health(baseUrl: String): ServerHealth
    fun classes(baseUrl: String): ClassCatalog
    fun predict(imageUri: Uri, baseUrl: String): LeafDiagnosis
}
