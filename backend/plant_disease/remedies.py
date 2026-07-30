"""Cultural-practice guidance for demo/API responses — not a treatment prescription."""

REMEDIES = {
    "Apple___Apple_scab": (
        "Remove and destroy fallen leaves and infected fruit each season. "
        "Prune for airflow. Resistant varieties help long-term."
    ),
    "Apple___Black_rot": (
        "Prune out dead or cankered wood and remove mummified fruit. "
        "Avoid wounding the tree during care."
    ),
    "Apple___Cedar_apple_rust": (
        "Remove nearby juniper/cedar hosts if practical. Prune infected leaves and twigs."
    ),
    "Apple___healthy": "No treatment needed — keep up regular pruning and monitoring.",
    "Blueberry___healthy": "No treatment needed — keep up regular monitoring.",
    "Cherry_(including_sour)___Powdery_mildew": (
        "Improve air circulation via pruning. Avoid excess nitrogen fertilizer. "
        "Remove infected shoots."
    ),
    "Cherry_(including_sour)___healthy": "No treatment needed.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (
        "Rotate away from corn for 1-2 seasons. Avoid overly dense planting. "
        "Resistant hybrids help."
    ),
    "Corn_(maize)___Common_rust_": (
        "Most modern hybrids tolerate mild infection. Monitor; consider resistant "
        "varieties next season."
    ),
    "Corn_(maize)___Northern_Leaf_Blight": (
        "Rotate crops and till under residue after harvest. Resistant hybrids are "
        "the most effective long-term fix."
    ),
    "Corn_(maize)___healthy": "No treatment needed.",
    "Grape___Black_rot": (
        "Remove mummified berries and infected canes during dormant pruning. "
        "Improve canopy airflow."
    ),
    "Grape___Esca_(Black_Measles)": (
        "Remove and destroy infected wood. Avoid large pruning wounds in wet weather "
        "— there's no reliable cure once established, so prevention matters most."
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (
        "Remove infected leaves and improve canopy ventilation. Avoid overhead irrigation."
    ),
    "Grape___healthy": "No treatment needed.",
    "Orange___Haunglongbing_(Citrus_greening)": (
        "No cure currently exists. Remove and destroy infected trees to slow spread, "
        "and control psyllid insect vectors. Often reportable — contact your local "
        "agricultural authority."
    ),
    "Peach___Bacterial_spot": (
        "Avoid overhead irrigation and working around wet trees. Prune for airflow."
    ),
    "Peach___healthy": "No treatment needed.",
    "Pepper,_bell___Bacterial_spot": (
        "Use disease-free seed/transplants. Avoid overhead watering. Rotate crops."
    ),
    "Pepper,_bell___healthy": "No treatment needed.",
    "Potato___Early_blight": (
        "Rotate crops, remove infected foliage, avoid overhead watering, "
        "and ensure good plant spacing."
    ),
    "Potato___Late_blight": (
        "Remove and destroy infected plants promptly — this disease spreads fast. "
        "Avoid overhead irrigation and monitor closely in cool, wet weather."
    ),
    "Potato___healthy": "No treatment needed.",
    "Raspberry___healthy": "No treatment needed.",
    "Soybean___healthy": "No treatment needed.",
    "Squash___Powdery_mildew": (
        "Improve airflow via spacing/pruning. Avoid overhead watering. "
        "Remove heavily infected leaves."
    ),
    "Strawberry___Leaf_scorch": (
        "Remove infected leaves after harvest. Avoid overhead irrigation. "
        "Renovate beds annually."
    ),
    "Strawberry___healthy": "No treatment needed.",
    "Tomato___Bacterial_spot": (
        "Use disease-free seed, avoid overhead watering, avoid handling wet plants, "
        "and rotate crops."
    ),
    "Tomato___Early_blight": (
        "Remove lower infected leaves, mulch to reduce soil splash onto foliage, "
        "and rotate crops."
    ),
    "Tomato___Late_blight": (
        "Remove and destroy infected plants promptly. Avoid overhead watering and "
        "monitor closely during cool, wet weather."
    ),
    "Tomato___Leaf_Mold": (
        "Improve ventilation and reduce humidity around foliage. Avoid overhead watering."
    ),
    "Tomato___Septoria_leaf_spot": (
        "Remove infected lower leaves, mulch, avoid overhead watering, and rotate crops."
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Mites favor dry conditions — occasional foliage rinsing and natural predators can help."
    ),
    "Tomato___Target_Spot": (
        "Remove infected foliage, improve airflow, and rotate crops."
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Control whitefly vectors, remove and destroy infected plants, and use "
        "resistant varieties where available."
    ),
    "Tomato___Tomato_mosaic_virus": (
        "Remove and destroy infected plants. Wash hands/tools between plants — "
        "it spreads easily by contact."
    ),
    "Tomato___healthy": "No treatment needed.",
}

DEFAULT_REMEDY = "No remedy info available for this class yet."


def remedy_for(class_name: str) -> str:
    return REMEDIES.get(class_name, DEFAULT_REMEDY)
