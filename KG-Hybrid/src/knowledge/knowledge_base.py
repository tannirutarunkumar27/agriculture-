"""
Master Agricultural Knowledge Base Registry
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation

Contains manually specified 4-point cardinal bounds profiles for all 22 crops:
apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute,
kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange,
papaya, pigeonpeas, pomegranate, rice, watermelon.

Institutional guidance is declared, but per-cell document/page/range provenance
is not retained here. External derivation and independence from the benchmark
have not been established by source-level evidence. These cardinal bounds are
a separate representation from the active fusion matrix's manual weights.
"""

from typing import Dict
from src.knowledge.knowledge_schema import (
    CardinalBounds,
    CropKnowledgeProfile,
    SourceType
)
from src.knowledge.knowledge_sources import register_source

KNOWLEDGE_BASE: Dict[str, CropKnowledgeProfile] = {
    "apple": CropKnowledgeProfile(
        crop_name="apple",
        N_bounds=CardinalBounds(10.0, 20.0, 40.0, 60.0),
        P_bounds=CardinalBounds(10.0, 120.0, 145.0, 160.0),
        K_bounds=CardinalBounds(150.0, 195.0, 205.0, 220.0),
        temp_bounds=CardinalBounds(15.0, 21.0, 24.0, 30.0),
        humidity_bounds=CardinalBounds(80.0, 90.0, 95.0, 100.0),
        ph_bounds=CardinalBounds(5.5, 5.8, 6.5, 7.8),
        rainfall_bounds=CardinalBounds(100.0, 105.0, 125.0, 150.0),
        growing_seasons=["Temperate Winter"],
        preferred_soil_types=["Deep Loamy", "Well-Drained Hill Soil"]
    ),
    "banana": CropKnowledgeProfile(
        crop_name="banana",
        N_bounds=CardinalBounds(80.0, 90.0, 110.0, 130.0),
        P_bounds=CardinalBounds(70.0, 75.0, 95.0, 110.0),
        K_bounds=CardinalBounds(45.0, 48.0, 55.0, 65.0),
        temp_bounds=CardinalBounds(25.0, 27.0, 29.0, 35.0),
        humidity_bounds=CardinalBounds(75.0, 78.0, 85.0, 95.0),
        ph_bounds=CardinalBounds(5.5, 5.7, 6.5, 7.5),
        rainfall_bounds=CardinalBounds(90.0, 95.0, 115.0, 130.0),
        growing_seasons=["Year-Round Tropical"],
        preferred_soil_types=["Rich Alluvial Loam", "Clay Loam"]
    ),
    "blackgram": CropKnowledgeProfile(
        crop_name="blackgram",
        N_bounds=CardinalBounds(30.0, 35.0, 45.0, 60.0),
        P_bounds=CardinalBounds(55.0, 60.0, 75.0, 85.0),
        K_bounds=CardinalBounds(15.0, 18.0, 25.0, 35.0),
        temp_bounds=CardinalBounds(25.0, 27.0, 32.0, 38.0),
        humidity_bounds=CardinalBounds(60.0, 62.0, 68.0, 75.0),
        ph_bounds=CardinalBounds(6.0, 6.5, 7.2, 7.8),
        rainfall_bounds=CardinalBounds(60.0, 62.0, 72.0, 85.0),
        growing_seasons=["Kharif", "Rabi"],
        preferred_soil_types=["Loamy", "Black Cotton Soil"]
    ),
    "chickpea": CropKnowledgeProfile(
        crop_name="chickpea",
        N_bounds=CardinalBounds(30.0, 38.0, 45.0, 60.0),
        P_bounds=CardinalBounds(55.0, 65.0, 75.0, 90.0),
        K_bounds=CardinalBounds(75.0, 78.0, 85.0, 95.0),
        temp_bounds=CardinalBounds(17.0, 18.5, 20.0, 25.0),
        humidity_bounds=CardinalBounds(10.0, 14.0, 18.0, 22.0),
        ph_bounds=CardinalBounds(5.5, 6.0, 7.3, 8.5),
        rainfall_bounds=CardinalBounds(60.0, 65.0, 85.0, 95.0),
        growing_seasons=["Rabi"],
        preferred_soil_types=["Deep Sandy Loam", "Clay Loam"]
    ),
    "coconut": CropKnowledgeProfile(
        crop_name="coconut",
        N_bounds=CardinalBounds(15.0, 20.0, 25.0, 40.0),
        P_bounds=CardinalBounds(5.0, 10.0, 25.0, 35.0),
        K_bounds=CardinalBounds(25.0, 28.0, 35.0, 45.0),
        temp_bounds=CardinalBounds(25.0, 26.0, 28.0, 32.0),
        humidity_bounds=CardinalBounds(90.0, 92.0, 98.0, 100.0),
        ph_bounds=CardinalBounds(5.0, 5.5, 6.5, 8.0),
        rainfall_bounds=CardinalBounds(130.0, 140.0, 175.0, 220.0),
        growing_seasons=["Perennial Coastal"],
        preferred_soil_types=["Coastal Alluvium", "Laterite"]
    ),
    "coffee": CropKnowledgeProfile(
        crop_name="coffee",
        N_bounds=CardinalBounds(80.0, 100.0, 110.0, 130.0),
        P_bounds=CardinalBounds(15.0, 20.0, 35.0, 45.0),
        K_bounds=CardinalBounds(25.0, 28.0, 35.0, 45.0),
        temp_bounds=CardinalBounds(23.0, 24.0, 27.0, 30.0),
        humidity_bounds=CardinalBounds(50.0, 52.0, 60.0, 70.0),
        ph_bounds=CardinalBounds(6.0, 6.2, 6.8, 7.5),
        rainfall_bounds=CardinalBounds(115.0, 120.0, 160.0, 200.0),
        growing_seasons=["Perennial Hill"],
        preferred_soil_types=["Deep Red Hill Soil", "Volcanic Loam"]
    ),
    "cotton": CropKnowledgeProfile(
        crop_name="cotton",
        N_bounds=CardinalBounds(100.0, 115.0, 125.0, 140.0),
        P_bounds=CardinalBounds(35.0, 40.0, 55.0, 70.0),
        K_bounds=CardinalBounds(15.0, 18.0, 22.0, 30.0),
        temp_bounds=CardinalBounds(22.0, 23.0, 26.0, 30.0),
        humidity_bounds=CardinalBounds(75.0, 78.0, 82.0, 90.0),
        ph_bounds=CardinalBounds(6.0, 6.3, 7.5, 8.0),
        rainfall_bounds=CardinalBounds(60.0, 65.0, 85.0, 100.0),
        growing_seasons=["Kharif"],
        preferred_soil_types=["Deep Black Cotton Soil (Vertisol)"]
    ),
    "grapes": CropKnowledgeProfile(
        crop_name="grapes",
        N_bounds=CardinalBounds(0.0, 20.0, 30.0, 40.0),
        P_bounds=CardinalBounds(120.0, 125.0, 140.0, 150.0),
        K_bounds=CardinalBounds(195.0, 198.0, 205.0, 215.0),
        temp_bounds=CardinalBounds(8.0, 11.0, 30.0, 40.0),
        humidity_bounds=CardinalBounds(80.0, 81.0, 83.0, 85.0),
        ph_bounds=CardinalBounds(5.5, 5.7, 6.5, 7.0),
        rainfall_bounds=CardinalBounds(65.0, 66.0, 72.0, 80.0),
        growing_seasons=["Sub-Tropical / Temperate"],
        preferred_soil_types=["Sandy Loam", "Gravelly Soil"]
    ),
    "jute": CropKnowledgeProfile(
        crop_name="jute",
        N_bounds=CardinalBounds(60.0, 75.0, 85.0, 100.0),
        P_bounds=CardinalBounds(35.0, 42.0, 55.0, 65.0),
        K_bounds=CardinalBounds(35.0, 38.0, 45.0, 55.0),
        temp_bounds=CardinalBounds(23.0, 24.5, 26.0, 30.0),
        humidity_bounds=CardinalBounds(70.0, 72.0, 78.0, 85.0),
        ph_bounds=CardinalBounds(6.0, 6.2, 7.0, 7.8),
        rainfall_bounds=CardinalBounds(150.0, 160.0, 185.0, 210.0),
        growing_seasons=["Kharif / Pre-Monsoon"],
        preferred_soil_types=["Alluvial Deltaic Loam"]
    ),
    "kidneybeans": CropKnowledgeProfile(
        crop_name="kidneybeans",
        N_bounds=CardinalBounds(15.0, 20.0, 25.0, 40.0),
        P_bounds=CardinalBounds(55.0, 60.0, 75.0, 90.0),
        K_bounds=CardinalBounds(15.0, 18.0, 22.0, 30.0),
        temp_bounds=CardinalBounds(15.0, 18.0, 24.0, 30.0),
        humidity_bounds=CardinalBounds(18.0, 20.0, 24.0, 30.0),
        ph_bounds=CardinalBounds(5.5, 5.7, 5.9, 6.5),
        rainfall_bounds=CardinalBounds(60.0, 95.0, 125.0, 160.0),
        growing_seasons=["Rabi / Cool Climate"],
        preferred_soil_types=["Light Sandy Loam"]
    ),
    "lentil": CropKnowledgeProfile(
        crop_name="lentil",
        N_bounds=CardinalBounds(10.0, 18.0, 25.0, 40.0),
        P_bounds=CardinalBounds(55.0, 65.0, 75.0, 90.0),
        K_bounds=CardinalBounds(15.0, 18.0, 22.0, 30.0),
        temp_bounds=CardinalBounds(18.0, 22.0, 28.0, 32.0),
        humidity_bounds=CardinalBounds(60.0, 62.0, 68.0, 75.0),
        ph_bounds=CardinalBounds(6.0, 6.3, 7.2, 7.8),
        rainfall_bounds=CardinalBounds(35.0, 40.0, 50.0, 60.0),
        growing_seasons=["Rabi"],
        preferred_soil_types=["Alluvial Loam", "Silt Loam"]
    ),
    "maize": CropKnowledgeProfile(
        crop_name="maize",
        N_bounds=CardinalBounds(70.0, 78.0, 95.0, 110.0),
        P_bounds=CardinalBounds(35.0, 42.0, 55.0, 70.0),
        K_bounds=CardinalBounds(15.0, 18.0, 22.0, 30.0),
        temp_bounds=CardinalBounds(18.0, 20.0, 25.0, 30.0),
        humidity_bounds=CardinalBounds(55.0, 58.0, 72.0, 85.0),
        ph_bounds=CardinalBounds(5.5, 5.8, 6.5, 7.5),
        rainfall_bounds=CardinalBounds(60.0, 75.0, 95.0, 115.0),
        growing_seasons=["Kharif", "Rabi"],
        preferred_soil_types=["Well-Drained Fertile Loam"]
    ),
    "mango": CropKnowledgeProfile(
        crop_name="mango",
        N_bounds=CardinalBounds(10.0, 20.0, 30.0, 40.0),
        P_bounds=CardinalBounds(15.0, 20.0, 35.0, 45.0),
        K_bounds=CardinalBounds(25.0, 28.0, 35.0, 45.0),
        temp_bounds=CardinalBounds(27.0, 29.0, 33.0, 38.0),
        humidity_bounds=CardinalBounds(45.0, 48.0, 53.0, 60.0),
        ph_bounds=CardinalBounds(4.5, 5.0, 6.5, 7.0),
        rainfall_bounds=CardinalBounds(85.0, 90.0, 98.0, 105.0),
        growing_seasons=["Perennial Tropical"],
        preferred_soil_types=["Deep Alluvial", "Lateritic Soil"]
    ),
    "mothbeans": CropKnowledgeProfile(
        crop_name="mothbeans",
        N_bounds=CardinalBounds(0.0, 15.0, 25.0, 40.0),
        P_bounds=CardinalBounds(35.0, 42.0, 55.0, 70.0),
        K_bounds=CardinalBounds(15.0, 18.0, 22.0, 30.0),
        temp_bounds=CardinalBounds(24.0, 27.0, 30.0, 35.0),
        humidity_bounds=CardinalBounds(45.0, 50.0, 60.0, 70.0),
        ph_bounds=CardinalBounds(3.5, 4.5, 9.0, 10.0),
        rainfall_bounds=CardinalBounds(30.0, 40.0, 60.0, 80.0),
        growing_seasons=["Kharif Arid"],
        preferred_soil_types=["Arid Sandy Soil"]
    ),
    "mungbean": CropKnowledgeProfile(
        crop_name="mungbean",
        N_bounds=CardinalBounds(0.0, 18.0, 25.0, 40.0),
        P_bounds=CardinalBounds(35.0, 42.0, 55.0, 70.0),
        K_bounds=CardinalBounds(15.0, 18.0, 22.0, 30.0),
        temp_bounds=CardinalBounds(27.0, 28.0, 29.0, 32.0),
        humidity_bounds=CardinalBounds(80.0, 82.0, 88.0, 95.0),
        ph_bounds=CardinalBounds(6.0, 6.2, 6.8, 7.5),
        rainfall_bounds=CardinalBounds(35.0, 40.0, 55.0, 70.0),
        growing_seasons=["Kharif", "Summer"],
        preferred_soil_types=["Well-Drained Loam"]
    ),
    "muskmelon": CropKnowledgeProfile(
        crop_name="muskmelon",
        N_bounds=CardinalBounds(80.0, 95.0, 115.0, 130.0),
        P_bounds=CardinalBounds(5.0, 10.0, 25.0, 35.0),
        K_bounds=CardinalBounds(45.0, 48.0, 52.0, 60.0),
        temp_bounds=CardinalBounds(27.0, 28.0, 29.0, 32.0),
        humidity_bounds=CardinalBounds(85.0, 90.0, 94.0, 98.0),
        ph_bounds=CardinalBounds(6.0, 6.2, 6.5, 7.0),
        rainfall_bounds=CardinalBounds(20.0, 22.0, 27.0, 35.0),
        growing_seasons=["Summer / Zaid"],
        preferred_soil_types=["Sandy Loam", "Riverbed Soil"]
    ),
    "orange": CropKnowledgeProfile(
        crop_name="orange",
        N_bounds=CardinalBounds(10.0, 20.0, 30.0, 40.0),
        P_bounds=CardinalBounds(5.0, 10.0, 25.0, 35.0),
        K_bounds=CardinalBounds(5.0, 8.0, 12.0, 20.0),
        temp_bounds=CardinalBounds(10.0, 15.0, 30.0, 35.0),
        humidity_bounds=CardinalBounds(85.0, 90.0, 93.0, 98.0),
        ph_bounds=CardinalBounds(6.0, 6.5, 7.5, 8.5),
        rainfall_bounds=CardinalBounds(100.0, 105.0, 115.0, 125.0),
        growing_seasons=["Sub-Tropical Perennial"],
        preferred_soil_types=["Well-Drained Sub-Tropical Loam"]
    ),
    "papaya": CropKnowledgeProfile(
        crop_name="papaya",
        N_bounds=CardinalBounds(35.0, 45.0, 55.0, 70.0),
        P_bounds=CardinalBounds(45.0, 52.0, 65.0, 80.0),
        K_bounds=CardinalBounds(45.0, 48.0, 52.0, 60.0),
        temp_bounds=CardinalBounds(23.0, 30.0, 38.0, 44.0),
        humidity_bounds=CardinalBounds(90.0, 92.0, 94.0, 98.0),
        ph_bounds=CardinalBounds(6.5, 6.6, 6.8, 7.2),
        rainfall_bounds=CardinalBounds(40.0, 95.0, 180.0, 250.0),
        growing_seasons=["Tropical Perennial"],
        preferred_soil_types=["Deep Rich Loam"]
    ),
    "pigeonpeas": CropKnowledgeProfile(
        crop_name="pigeonpeas",
        N_bounds=CardinalBounds(0.0, 15.0, 25.0, 40.0),
        P_bounds=CardinalBounds(55.0, 60.0, 75.0, 90.0),
        K_bounds=CardinalBounds(15.0, 18.0, 22.0, 30.0),
        temp_bounds=CardinalBounds(18.0, 20.0, 30.0, 38.0),
        humidity_bounds=CardinalBounds(40.0, 45.0, 60.0, 75.0),
        ph_bounds=CardinalBounds(4.5, 5.0, 6.8, 7.5),
        rainfall_bounds=CardinalBounds(90.0, 100.0, 160.0, 220.0),
        growing_seasons=["Kharif"],
        preferred_soil_types=["Deep Black & Red Loam"]
    ),
    "pomegranate": CropKnowledgeProfile(
        crop_name="pomegranate",
        N_bounds=CardinalBounds(0.0, 15.0, 25.0, 40.0),
        P_bounds=CardinalBounds(5.0, 10.0, 25.0, 35.0),
        K_bounds=CardinalBounds(35.0, 38.0, 42.0, 50.0),
        temp_bounds=CardinalBounds(18.0, 20.0, 24.0, 30.0),
        humidity_bounds=CardinalBounds(85.0, 88.0, 92.0, 95.0),
        ph_bounds=CardinalBounds(5.5, 6.0, 6.8, 7.5),
        rainfall_bounds=CardinalBounds(100.0, 105.0, 112.0, 120.0),
        growing_seasons=["Arid / Semi-Arid"],
        preferred_soil_types=["Deep Heavy Loam", "Gravelly Soil"]
    ),
    "rice": CropKnowledgeProfile(
        crop_name="rice",
        N_bounds=CardinalBounds(60.0, 80.0, 100.0, 120.0),
        P_bounds=CardinalBounds(35.0, 40.0, 55.0, 70.0),
        K_bounds=CardinalBounds(35.0, 38.0, 42.0, 50.0),
        temp_bounds=CardinalBounds(20.0, 22.0, 27.0, 32.0),
        humidity_bounds=CardinalBounds(75.0, 80.0, 85.0, 95.0),
        ph_bounds=CardinalBounds(5.0, 5.5, 6.8, 7.8),
        rainfall_bounds=CardinalBounds(180.0, 200.0, 250.0, 300.0),
        growing_seasons=["Kharif", "Rabi", "Summer (Virippu, Mundakan, Puncha)"],
        preferred_soil_types=["Clayey Loam", "Alluvial Silt"]
    ),
    "watermelon": CropKnowledgeProfile(
        crop_name="watermelon",
        N_bounds=CardinalBounds(80.0, 95.0, 110.0, 130.0),
        P_bounds=CardinalBounds(5.0, 10.0, 25.0, 35.0),
        K_bounds=CardinalBounds(45.0, 48.0, 52.0, 60.0),
        temp_bounds=CardinalBounds(24.0, 25.0, 27.0, 30.0),
        humidity_bounds=CardinalBounds(80.0, 82.0, 88.0, 95.0),
        ph_bounds=CardinalBounds(6.0, 6.2, 6.8, 7.5),
        rainfall_bounds=CardinalBounds(40.0, 45.0, 55.0, 70.0),
        growing_seasons=["Summer / Zaid"],
        preferred_soil_types=["Deep Sandy Loam", "Alluvial Riverbed"]
    )
}

# Ordered class names list matching target label encoding index 0..21
CROP_CLASSES_ORDERED = sorted(list(KNOWLEDGE_BASE.keys()))
