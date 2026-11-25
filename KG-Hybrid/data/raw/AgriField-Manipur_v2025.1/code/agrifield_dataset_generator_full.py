# agrifield_dataset_generator_full.py
"""
Full AgriField-Manipur dataset generator (final, realistic).
Outputs:
 - data/ml_dataset.csv        -> sensor features + crop label (ML-ready)
 - data/final_dataset.csv     -> enriched rows with advisories (deployment-ready)
Also writes base files:
 - data/fields_sample.csv
 - data/soil_by_district.csv
 - data/advisory_rules.yaml
 - data/config.yaml
"""

import os
import csv
import random
import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml

# ----------------------- USER CONFIG -----------------------
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE = "2025-06-01"
END_DATE   = "2025-10-31"   # user requested
FIELDS_PER_DISTRICT = 20    # 20 × 5 = 100 fields
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# districts and approximate centers (latitude, longitude)
DISTRICT_CENTERS = {
    "Imphal East": (24.82, 93.93),
    "Imphal West": (24.79, 93.93),
    "Thoubal": (24.71, 93.91),
    "Bishnupur": (24.66, 93.96),
    "Kakching": (24.57, 93.87)
}

# Finalized 50 crops for Manipur valley (as agreed)
CROPS = [
    "Rice","Maize","Finger millet","Pearl millet","Tomato","Brinjal","Cabbage",
    "Cauliflower","Carrot","Radish","French bean","Pea","Okra","Spinach","Lettuce",
    "Pumpkin","Bottle gourd","Ridge gourd","Bitter gourd","King Chilli","Banana",
    "Orange","Pineapple","Papaya","Guava","Mango","Passion fruit","Strawberry",
    "Lemon","Jackfruit","Ginger","Turmeric","Garlic","Chilli (other)","Coriander",
    "Mint","Fenugreek","Amaranth","Mustard greens","Celosia","Soybean","Black gram",
    "Green gram","Chickpea","Cowpea","Lentil","Sesame","Groundnut","Potato",
    "Sweet potato","Yam","Taro"
]

# -------------------- district soil baselines (realistic approximations) --------------------
# These are district-average baselines we will perturb per field/day
SOIL_BY_DISTRICT = {
    "Imphal East":  {"pH":6.5, "N":85, "P":40, "K":42, "organic_carbon":1.2},
    "Imphal West":  {"pH":6.6, "N":80, "P":38, "K":40, "organic_carbon":1.1},
    "Thoubal":      {"pH":6.4, "N":82, "P":39, "K":41, "organic_carbon":1.0},
    "Bishnupur":    {"pH":6.3, "N":78, "P":36, "K":39, "organic_carbon":1.1},
    "Kakching":     {"pH":6.5, "N":80, "P":37, "K":40, "organic_carbon":1.2}
}

# -------------------- advisory rules per crop (realistic ranges) --------------------
# Each crop has:
#  temp_range [min,max] (°C)
#  humidity_range [min,max] (%)
#  pH_range [min,max]
#  soil_moisture_min (%) -> threshold for irrigation
#  NPK_required (g per application) -> generic weekly amount
#  irrigation_ml (ml per irrigation action)
#  fertilizer_interval_days (days cooldown)
#  pest_risk notes (basic)
ADVISORY_RULES = {
    # cereals
    "Rice":            {"temp_range":[22,32],"humidity_range":[70,90],"pH_range":[5.5,6.5],"soil_moisture_min":45,"NPK_required":30,"irrigation_ml":800,"fertilizer_interval_days":14,"pest_risk":{"stem_borer":"medium","leaf_blight":"medium"}},
    "Maize":           {"temp_range":[18,30],"humidity_range":[55,80],"pH_range":[5.8,6.8],"soil_moisture_min":35,"NPK_required":25,"irrigation_ml":600,"fertilizer_interval_days":10,"pest_risk":{"stem_borer":"medium"}},
    "Finger millet":   {"temp_range":[20,30],"humidity_range":[50,75],"pH_range":[5.5,7.0],"soil_moisture_min":30,"NPK_required":20,"irrigation_ml":400,"fertilizer_interval_days":14,"pest_risk":{"leaf_spot":"low"}},
    "Pearl millet":    {"temp_range":[22,33],"humidity_range":[30,60],"pH_range":[5.5,7.5],"soil_moisture_min":25,"NPK_required":18,"irrigation_ml":350,"fertilizer_interval_days":14,"pest_risk":{"stem_borer":"low"}},
    # vegetables
    "Tomato":          {"temp_range":[20,28],"humidity_range":[60,80],"pH_range":[6.0,6.8],"soil_moisture_min":30,"NPK_required":20,"irrigation_ml":450,"fertilizer_interval_days":7,"pest_risk":{"aphids":"medium","blight":"medium"}},
    "Brinjal":         {"temp_range":[22,30],"humidity_range":[60,80],"pH_range":[6.0,6.8],"soil_moisture_min":30,"NPK_required":18,"irrigation_ml":450,"fertilizer_interval_days":7,"pest_risk":{"fruit_borer":"medium"}},
    "Cabbage":         {"temp_range":[15,22],"humidity_range":[65,80],"pH_range":[6.0,7.0],"soil_moisture_min":35,"NPK_required":20,"irrigation_ml":500,"fertilizer_interval_days":10,"pest_risk":{"caterpillars":"medium"}},
    "Cauliflower":     {"temp_range":[15,22],"humidity_range":[60,80],"pH_range":[6.0,7.5],"soil_moisture_min":35,"NPK_required":20,"irrigation_ml":450,"fertilizer_interval_days":10,"pest_risk":{"caterpillars":"medium"}},
    "Carrot":          {"temp_range":[10,22],"humidity_range":[60,80],"pH_range":[6.0,6.8],"soil_moisture_min":30,"NPK_required":12,"irrigation_ml":350,"fertilizer_interval_days":14,"pest_risk":{"root_rot":"low"}},
    "Radish":          {"temp_range":[10,24],"humidity_range":[60,80],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":10,"irrigation_ml":300,"fertilizer_interval_days":14,"pest_risk":{"root_rot":"low"}},
    "French bean":     {"temp_range":[18,28],"humidity_range":[60,85],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":12,"irrigation_ml":300,"fertilizer_interval_days":10,"pest_risk":{"pod_borer":"medium"}},
    "Pea":             {"temp_range":[10,22],"humidity_range":[60,80],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":12,"irrigation_ml":300,"fertilizer_interval_days":10,"pest_risk":{"powdery_mildew":"low"}},
    "Okra":            {"temp_range":[20,32],"humidity_range":[60,85],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":15,"irrigation_ml":400,"fertilizer_interval_days":7,"pest_risk":{"jassids":"medium"}},
    "Spinach":         {"temp_range":[12,26],"humidity_range":[60,85],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":10,"irrigation_ml":300,"fertilizer_interval_days":7,"pest_risk":{"leaf_miner":"low"}},
    "Lettuce":         {"temp_range":[8,20],"humidity_range":[60,80],"pH_range":[6.0,7.0],"soil_moisture_min":35,"NPK_required":10,"irrigation_ml":300,"fertilizer_interval_days":10,"pest_risk":{"aphids":"medium"}},
    "Pumpkin":         {"temp_range":[20,30],"humidity_range":[60,80],"pH_range":[6.0,7.5],"soil_moisture_min":35,"NPK_required":20,"irrigation_ml":600,"fertilizer_interval_days":10,"pest_risk":{"powdery_mildew":"medium"}},
    "Bottle gourd":    {"temp_range":[20,30],"humidity_range":[60,80],"pH_range":[6.0,7.5],"soil_moisture_min":35,"NPK_required":15,"irrigation_ml":500,"fertilizer_interval_days":10,"pest_risk":{"viral":"medium"}},
    "Ridge gourd":     {"temp_range":[20,32],"humidity_range":[60,85],"pH_range":[6.0,7.5],"soil_moisture_min":35,"NPK_required":15,"irrigation_ml":500,"fertilizer_interval_days":10,"pest_risk":{"fruit_borer":"medium"}},
    "Bitter gourd":    {"temp_range":[22,30],"humidity_range":[65,85],"pH_range":[6.0,7.0],"soil_moisture_min":35,"NPK_required":18,"irrigation_ml":500,"fertilizer_interval_days":7,"pest_risk":{"viral":"medium"}},
    "King Chilli":     {"temp_range":[24,30],"humidity_range":[70,85],"pH_range":[6.0,7.0],"soil_moisture_min":35,"NPK_required":20,"irrigation_ml":500,"fertilizer_interval_days":7,"pest_risk":{"aphids":"low","mites":"medium"}},
    # fruits
    "Banana":          {"temp_range":[22,32],"humidity_range":[70,90],"pH_range":[5.5,7.0],"soil_moisture_min":40,"NPK_required":40,"irrigation_ml":800,"fertilizer_interval_days":14,"pest_risk":{"nematodes":"medium"}},
    "Orange":          {"temp_range":[18,30],"humidity_range":[55,80],"pH_range":[5.5,6.8],"soil_moisture_min":30,"NPK_required":30,"irrigation_ml":600,"fertilizer_interval_days":14,"pest_risk":{"citrus_borer":"medium"}},
    "Pineapple":       {"temp_range":[20,30],"humidity_range":[60,85],"pH_range":[4.5,6.5],"soil_moisture_min":30,"NPK_required":25,"irrigation_ml":400,"fertilizer_interval_days":21,"pest_risk":{"mealy_bug":"medium"}},
    "Papaya":          {"temp_range":[22,32],"humidity_range":[60,85],"pH_range":[5.5,7.0],"soil_moisture_min":35,"NPK_required":30,"irrigation_ml":600,"fertilizer_interval_days":14,"pest_risk":{"fruit_flies":"medium"}},
    "Guava":           {"temp_range":[20,30],"humidity_range":[55,80],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":25,"irrigation_ml":500,"fertilizer_interval_days":21,"pest_risk":{"fruit_borer":"medium"}},
    "Mango":           {"temp_range":[24,34],"humidity_range":[60,85],"pH_range":[5.5,7.5],"soil_moisture_min":40,"NPK_required":40,"irrigation_ml":700,"fertilizer_interval_days":21,"pest_risk":{"fruit_borer":"medium"}},
    "Passion fruit":   {"temp_range":[18,28],"humidity_range":[60,85],"pH_range":[5.5,6.8],"soil_moisture_min":35,"NPK_required":25,"irrigation_ml":500,"fertilizer_interval_days":14,"pest_risk":{"fungal":"medium"}},
    "Strawberry":      {"temp_range":[12,22],"humidity_range":[60,80],"pH_range":[5.5,6.5],"soil_moisture_min":35,"NPK_required":20,"irrigation_ml":300,"fertilizer_interval_days":14,"pest_risk":{"botrytis":"medium"}},
    "Lemon":           {"temp_range":[18,30],"humidity_range":[50,80],"pH_range":[5.5,6.5],"soil_moisture_min":30,"NPK_required":25,"irrigation_ml":400,"fertilizer_interval_days":21,"pest_risk":{"canker":"medium"}},
    "Jackfruit":       {"temp_range":[20,32],"humidity_range":[60,85],"pH_range":[6.0,7.5],"soil_moisture_min":40,"NPK_required":35,"irrigation_ml":700,"fertilizer_interval_days":21,"pest_risk":{"fruit_borer":"medium"}},
    # spices & herbs
    "Ginger":          {"temp_range":[20,30],"humidity_range":[60,85],"pH_range":[5.5,6.5],"soil_moisture_min":35,"NPK_required":25,"irrigation_ml":500,"fertilizer_interval_days":21,"pest_risk":{"nematode":"medium"}},
    "Turmeric":        {"temp_range":[20,30],"humidity_range":[60,85],"pH_range":[5.5,6.5],"soil_moisture_min":35,"NPK_required":25,"irrigation_ml":500,"fertilizer_interval_days":21,"pest_risk":{"root_rot":"medium"}},
    "Garlic":          {"temp_range":[12,24],"humidity_range":[55,75],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":15,"irrigation_ml":300,"fertilizer_interval_days":21,"pest_risk":{"thrips":"low"}},
    "Chilli (other)":  {"temp_range":[22,32],"humidity_range":[60,85],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":18,"irrigation_ml":400,"fertilizer_interval_days":7,"pest_risk":{"thrips":"medium"}},
    "Coriander":       {"temp_range":[12,28],"humidity_range":[60,85],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":8,"irrigation_ml":200,"fertilizer_interval_days":14,"pest_risk":{"aphids":"low"}},
    "Mint":            {"temp_range":[12,28],"humidity_range":[60,85],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":8,"irrigation_ml":200,"fertilizer_interval_days":14,"pest_risk":{"leaf_hopper":"low"}},
    "Fenugreek":       {"temp_range":[12,26],"humidity_range":[60,80],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":8,"irrigation_ml":200,"fertilizer_interval_days":14,"pest_risk":{"low":"low"}},
    "Amaranth":        {"temp_range":[18,30],"humidity_range":[60,85],"pH_range":[6.0,7.5],"soil_moisture_min":30,"NPK_required":12,"irrigation_ml":300,"fertilizer_interval_days":10,"pest_risk":{"leaf_miner":"low"}},
    "Mustard greens":  {"temp_range":[10,22],"humidity_range":[60,80],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":10,"irrigation_ml":300,"fertilizer_interval_days":14,"pest_risk":{"low":"low"}},
    "Celosia":         {"temp_range":[18,30],"humidity_range":[60,85],"pH_range":[6.0,7.5],"soil_moisture_min":30,"NPK_required":10,"irrigation_ml":300,"fertilizer_interval_days":10,"pest_risk":{"low":"low"}},
    # legumes, oilseeds, tubers
    "Soybean":         {"temp_range":[18,30],"humidity_range":[55,80],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":20,"irrigation_ml":400,"fertilizer_interval_days":14,"pest_risk":{"pod_borer":"medium"}},
    "Black gram":      {"temp_range":[20,32],"humidity_range":[55,80],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":15,"irrigation_ml":350,"fertilizer_interval_days":14,"pest_risk":{"low":"low"}},
    "Green gram":      {"temp_range":[20,32],"humidity_range":[55,80],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":15,"irrigation_ml":350,"fertilizer_interval_days":14,"pest_risk":{"low":"low"}},
    "Chickpea":        {"temp_range":[18,28],"humidity_range":[40,70],"pH_range":[6.0,7.5],"soil_moisture_min":25,"NPK_required":12,"irrigation_ml":300,"fertilizer_interval_days":21,"pest_risk":{"low":"low"}},
    "Cowpea":          {"temp_range":[20,32],"humidity_range":[55,80],"pH_range":[6.0,7.0],"soil_moisture_min":30,"NPK_required":12,"irrigation_ml":350,"fertilizer_interval_days":14,"pest_risk":{"pod_borer":"medium"}},
    "Lentil":          {"temp_range":[15,25],"humidity_range":[40,70],"pH_range":[6.0,7.5],"soil_moisture_min":25,"NPK_required":12,"irrigation_ml":300,"fertilizer_interval_days":21,"pest_risk":{"low":"low"}},
    "Sesame":          {"temp_range":[22,32],"humidity_range":[40,70],"pH_range":[6.0,7.5],"soil_moisture_min":25,"NPK_required":12,"irrigation_ml":300,"fertilizer_interval_days":21,"pest_risk":{"low":"low"}},
    "Groundnut":       {"temp_range":[22,32],"humidity_range":[60,85],"pH_range":[5.5,6.5],"soil_moisture_min":30,"NPK_required":20,"irrigation_ml":450,"fertilizer_interval_days":21,"pest_risk":{"nematode":"medium"}},
    "Potato":          {"temp_range":[12,22],"humidity_range":[60,80],"pH_range":[5.5,6.5],"soil_moisture_min":30,"NPK_required":20,"irrigation_ml":450,"fertilizer_interval_days":14,"pest_risk":{"late_blight":"medium"}},
    "Sweet potato":    {"temp_range":[20,30],"humidity_range":[60,85],"pH_range":[5.5,6.5],"soil_moisture_min":30,"NPK_required":15,"irrigation_ml":400,"fertilizer_interval_days":21,"pest_risk":{"low":"low"}},
    "Yam":             {"temp_range":[22,30],"humidity_range":[60,85],"pH_range":[5.5,6.5],"soil_moisture_min":30,"NPK_required":20,"irrigation_ml":500,"fertilizer_interval_days":21,"pest_risk":{"low":"low"}},
    "Taro":            {"temp_range":[22,30],"humidity_range":[70,90],"pH_range":[5.5,6.8],"soil_moisture_min":40,"NPK_required":20,"irrigation_ml":600,"fertilizer_interval_days":21,"pest_risk":{"root_rot":"medium"}},
}

# Save advisory rules to YAML for record
with open(os.path.join(OUT_DIR, "advisory_rules.yaml"), "w") as f:
    yaml.dump(ADVISORY_RULES, f, sort_keys=True)

# -------------------- write soil_by_district.csv --------------------
soil_rows = []
for dname, values in SOIL_BY_DISTRICT.items():
    soil_rows.append({
        "district": dname,
        "pH": values["pH"],
        "N": values["N"],
        "P": values["P"],
        "K": values["K"],
        "organic_carbon": values["organic_carbon"]
    })
pd.DataFrame(soil_rows).to_csv(os.path.join(OUT_DIR, "soil_by_district.csv"), index=False)

# -------------------- create fields_sample.csv (100 fields, 20 per district) --------------------
fields = []
crop_pool = CROPS.copy() * 2  # ensure enough variety
random.shuffle(crop_pool)
fid = 1
for dname, (lat0, lon0) in DISTRICT_CENTERS.items():
    for i in range(FIELDS_PER_DISTRICT):
        field_id = f"F{fid:03d}"
        # jitter lat/lon in small radius ~ +/- 0.015 deg (~1-1.5km)
        lat = round(lat0 + random.uniform(-0.015, 0.015), 6)
        lon = round(lon0 + random.uniform(-0.015, 0.015), 6)
        crop = crop_pool.pop() if crop_pool else random.choice(CROPS)
        fields.append({"field_id": field_id, "district": dname, "lat": lat, "lon": lon, "crop": crop})
        fid += 1
fields_df = pd.DataFrame(fields)
fields_df.to_csv(os.path.join(OUT_DIR, "fields_sample.csv"), index=False)

# -------------------- config save --------------------
config = {
    "fields_file": "fields_sample.csv",
    "soil_file": "soil_by_district.csv",
    "advisory_file": "advisory_rules.yaml",
    "output_ml": "ml_dataset.csv",
    "output_final": "final_dataset.csv",
    "start_date": START_DATE,
    "end_date": END_DATE,
    "fields_per_district": FIELDS_PER_DISTRICT
}
with open(os.path.join(OUT_DIR, "config.yaml"), "w") as f:
    yaml.dump(config, f)

# -------------------- helpers: seasonal patterns --------------------
def seasonal_temp(base=25.0, amplitude=3.0, day_of_year=180):
    return base + amplitude * math.sin(2*math.pi*(day_of_year/365.0))

def seasonal_humidity(base=70.0, amplitude=10.0, day_of_year=180):
    return base + amplitude * math.sin(2*math.pi*(day_of_year/365.0) - 0.5)

# date range
start = datetime.fromisoformat(START_DATE)
end   = datetime.fromisoformat(END_DATE)
dates = pd.date_range(start, end)

# -------------------- dataset generation --------------------
# We'll track last application dates so advisories don't repeat unnecessarily
last_fertilizer = {fid: None for fid in fields_df['field_id']}
last_irrigation = {fid: None for fid in fields_df['field_id']}

ml_rows = []
final_rows = []

soil_df = pd.read_csv(os.path.join(OUT_DIR, "soil_by_district.csv"))

for idx, field in fields_df.iterrows():
    fid = field['field_id']
    district = field['district']
    lat = field['lat']
    lon = field['lon']
    crop = field['crop']
    soil_base = SOIL_BY_DISTRICT[district]
    base_ph = float(soil_base["pH"])
    base_N = float(soil_base["N"])
    base_P = float(soil_base["P"])
    base_K = float(soil_base["K"])
    # pull crop rules
    rule = ADVISORY_RULES.get(crop)
    temp_min, temp_max = rule["temp_range"]
    hum_min, hum_max = rule["humidity_range"]
    pH_min, pH_max = rule["pH_range"]
    moisture_min = rule["soil_moisture_min"]
    npk_req = rule["NPK_required"]
    irrigation_ml = rule["irrigation_ml"]
    fert_interval = rule["fertilizer_interval_days"]
    pest_map = rule.get("pest_risk", {})
    
    # small field-level baseline adjustments (realistic): pH +/-0.2, NPK +/- small noise
    field_ph_baseline = round(base_ph + np.random.normal(scale=0.12), 2)
    field_N_base = max(5, int(base_N + np.random.normal(scale=4)))
    field_P_base = max(2, int(base_P + np.random.normal(scale=3)))
    field_K_base = max(10, int(base_K + np.random.normal(scale=4)))
    
    for d in dates:
        doy = d.timetuple().tm_yday
        # temperature: center around crop-preferred midpoint with seasonal signal
        base_mid = (temp_min + temp_max) / 2.0
        temp = seasonal_temp(base=base_mid, amplitude=3.0, day_of_year=doy) + np.random.normal(scale=1.1)
        # humidity: seasonal baseline around crop midpoint
        hum_mid = (hum_min + hum_max) / 2.0
        humidity = seasonal_humidity(base=hum_mid, amplitude=6.0, day_of_year=doy) + np.random.normal(scale=2.5)
        # rainfall: monsoon heavy June-Sep, taper Oct
        month = d.month
        if month in [6,7,8,9]:
            rainfall = float(round(max(0.0, np.random.gamma(shape=2.2, scale=6.0)),2))
        elif month == 10:
            rainfall = float(round(max(0.0, np.random.gamma(shape=1.0, scale=4.0)),2))
        else:
            rainfall = float(round(max(0.0, np.random.gamma(shape=0.6, scale=2.0)),2))
        # soil moisture: function of field baseline, rainfall, small noise
        soil_moisture = field_ph_baseline * 6.0 + np.random.normal(loc=0.0, scale=4.0)
        soil_moisture += min(30.0, rainfall * random.uniform(0.6,1.4))
        soil_moisture = float(round(max(5.0, min(95.0, soil_moisture + np.random.normal(scale=3.0))), 1))
        # pH variation around field baseline
        ph = float(round(field_ph_baseline + np.random.normal(scale=0.12), 2))
        # NPK variation (slow changes) — simulate small per-day noise
        N = int(max(2, round(field_N_base + np.random.normal(scale=2.5))))
        P = int(max(1, round(field_P_base + np.random.normal(scale=1.8))))
        K = int(max(5, round(field_K_base + np.random.normal(scale=2.5))))
        
        # ---------- Suitability check ----------
        checks = 0
        total = 4
        if temp_min <= temp <= temp_max: checks += 1
        if hum_min <= humidity <= hum_max: checks += 1
        if pH_min <= ph <= pH_max: checks += 1
        if soil_moisture >= moisture_min: checks += 1
        crop_suitable = "Yes" if (checks / total) >= 0.75 else "No"
        
        # ---------- Irrigation advisory ----------
        irrigation_advice = "No"
        if soil_moisture < moisture_min and rainfall < 12:
            last_i = last_irrigation.get(fid)
            # only advise if not advised within last 1 day (stateful)
            if (last_i is None) or ((d - last_i).days >= 1):
                irrigation_advice = f"Irrigate {irrigation_ml} ml"
                last_irrigation[fid] = d
        
        # ---------- Fertilizer advisory ----------
        fertilizer_advice = "No"
        # quick heuristic: if N < half expected annual (npk_req*2) or P,K low -> advise
        nutrient_deficit = (N < npk_req*1.5) or (P < max(2, int(npk_req*0.8))) or (K < npk_req*1.2)
        last_f = last_fertilizer.get(fid)
        if nutrient_deficit and ((last_f is None) or ((d - last_f).days >= fert_interval)):
            fertilizer_advice = f"Apply NPK {npk_req} g"
            last_fertilizer[fid] = d
        
        # ---------- Pest risk (rule-based) ----------
        pest_risk = "Low"
        # raise risk under warm+humid conditions or heavy rainfall
        if humidity > 82 and temp > (temp_max - 1):
            pest_risk = "High"
        elif humidity > 76 or rainfall > 20:
            pest_risk = "Medium"
        # ensure at least the level suggested by crop-specific pest_map
        if isinstance(pest_map, dict):
            # if any specific pest has "medium"/"high", escalate accordingly (conservative)
            if any(v == "high" for v in pest_map.values()):
                if pest_risk == "Low": pest_risk = "Medium"
            if any(v == "medium" for v in pest_map.values()):
                if pest_risk == "Low": pest_risk = "Medium"
        
        # ---------- rows ----------
        ml_row = {
            "date": d.strftime("%Y-%m-%d"),
            "field_id": fid,
            "district": district,
            "lat": lat,
            "lon": lon,
            "crop": crop,
            "temperature": round(float(temp),2),
            "humidity": round(float(humidity),2),
            "soil_moisture": soil_moisture,
            "ph": ph,
            "N": N,
            "P": P,
            "K": K,
            "rainfall": rainfall
        }
        final_row = ml_row.copy()
        final_row.update({
            "crop_suitable": crop_suitable,
            "irrigation_advice": irrigation_advice,
            "fertilizer_advice": fertilizer_advice,
            "pest_risk": pest_risk
        })
        ml_rows.append(ml_row)
        final_rows.append(final_row)

# -------------------- write outputs --------------------
ml_df = pd.DataFrame(ml_rows)
final_df = pd.DataFrame(final_rows)

ml_out = os.path.join(OUT_DIR, "ml_dataset.csv")
final_out = os.path.join(OUT_DIR, "final_dataset.csv")
ml_df.to_csv(ml_out, index=False)
final_df.to_csv(final_out, index=False)

print("✅ Done. Files written to:", OUT_DIR)
print(" - ML dataset:", ml_out, "rows:", len(ml_df))
print(" - Final dataset:", final_out, "rows:", len(final_df))
