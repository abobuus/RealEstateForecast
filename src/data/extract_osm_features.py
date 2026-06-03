"""
Experiment 3: Extract OSM infrastructure features for Moscow, St. Petersburg, Novosibirsk.

Steps:
1. Extract POIs (metro, schools, hospitals, parks) from PBF files
2. Filter apartments dataset to 3 cities (region 77, 78, 54)
3. Compute min distances to each POI type using KD-tree
4. Save result to data/processed/cities_3_with_infra.csv
"""

import os
import json
import numpy as np
import pandas as pd
import osmium
from scipy.spatial import cKDTree

DATA_DIR = "data"
OSM_DIR  = os.path.join(DATA_DIR, "osm")

# Region codes and corresponding OSM PBF files + bounding boxes
CITIES = {
    77: {
        "name": "Moscow",
        "pbf": os.path.join(OSM_DIR, "central-fed-district-260526.osm.pbf"),
        "bbox": (55.10, 36.70, 56.20, 38.40),  # (lat_min, lon_min, lat_max, lon_max)
    },
    78: {
        "name": "Saint Petersburg",
        "pbf": os.path.join(OSM_DIR, "northwestern-fed-district-260526.osm.pbf"),
        "bbox": (59.60, 29.50, 60.30, 31.00),
    },
    54: {
        "name": "Novosibirsk",
        "pbf": os.path.join(OSM_DIR, "siberian-fed-district-260526.osm.pbf"),
        "bbox": (54.55, 82.50, 55.25, 83.35),
    },
}

# POI extraction rules: name -> list of (tag_key, tag_value) conditions (AND for multiple)
POI_RULES = {
    "metro": [("railway", "station"), ("station", "subway")],
    "school": [("amenity", "school")],
    "hospital": None,   # special: amenity=hospital OR amenity=clinic
    "park": [("leisure", "park")],
}


def _in_bbox(lat, lon, bbox):
    lat_min, lon_min, lat_max, lon_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


class POIHandler(osmium.SimpleHandler):
    """Collects POI node/way centroids matching given rules."""

    def __init__(self, bbox):
        super().__init__()
        self.bbox = bbox
        self.results = {k: [] for k in POI_RULES}

    def _classify(self, tags, lat, lon):
        if not _in_bbox(lat, lon, self.bbox):
            return
        t = dict(tags)

        # Metro: railway=station AND station=subway
        if t.get("railway") == "station" and t.get("station") == "subway":
            self.results["metro"].append((lat, lon))

        # School
        if t.get("amenity") == "school":
            self.results["school"].append((lat, lon))

        # Hospital OR clinic
        if t.get("amenity") in ("hospital", "clinic"):
            self.results["hospital"].append((lat, lon))

        # Park
        if t.get("leisure") == "park":
            self.results["park"].append((lat, lon))

    def node(self, n):
        if n.location.valid():
            self._classify(n.tags, n.location.lat, n.location.lon)

    def way(self, w):
        # Compute way centroid from member nodes (for parks, hospital buildings, etc.)
        try:
            lats = [nd.location.lat for nd in w.nodes if nd.location.valid()]
            lons = [nd.location.lon for nd in w.nodes if nd.location.valid()]
            if lats:
                self._classify(w.tags, sum(lats) / len(lats), sum(lons) / len(lons))
        except Exception:
            pass


def extract_pois(city_key: int) -> dict[str, np.ndarray]:
    """Extract POIs for a city from its PBF file. Returns dict of arrays (N, 2)."""
    cfg = CITIES[city_key]
    cache_path = os.path.join(OSM_DIR, f"pois_{cfg['name'].lower().replace(' ', '_')}.json")

    if os.path.exists(cache_path):
        print(f"  Loading POIs from cache: {cache_path}")
        with open(cache_path) as f:
            raw = json.load(f)
        return {k: np.array(v) for k, v in raw.items()}

    print(f"  Scanning {cfg['pbf']} (may take 5-20 min)...")
    handler = POIHandler(cfg["bbox"])
    handler.apply_file(cfg["pbf"], locations=True, idx="flex_mem")

    pois = {}
    for k, pts in handler.results.items():
        arr = np.array(pts) if pts else np.empty((0, 2))
        pois[k] = arr
        print(f"    {k}: {len(arr)} objects")

    # Save cache
    with open(cache_path, "w") as f:
        json.dump({k: v.tolist() for k, v in pois.items()}, f)
    print(f"  Saved to {cache_path}")
    return pois


def haversine_approx_km(lat_center: float) -> float:
    """Returns metres-per-degree-lat and metres-per-degree-lon at given latitude."""
    R = 6371.0
    lat_rad = np.radians(lat_center)
    km_per_lat = np.pi * R / 180.0
    km_per_lon = km_per_lat * np.cos(lat_rad)
    return km_per_lat, km_per_lon


def compute_distances(df: pd.DataFrame, pois: dict[str, np.ndarray], lat_center: float) -> pd.DataFrame:
    """Add dist_metro, dist_school, dist_hospital, dist_park columns (km)."""
    km_per_lat, km_per_lon = haversine_approx_km(lat_center)
    pts_apt = df[["geo_lat", "geo_lon"]].values.copy()
    pts_apt[:, 0] *= km_per_lat
    pts_apt[:, 1] *= km_per_lon

    for poi_type, poi_arr in pois.items():
        col = f"dist_{poi_type}"
        if len(poi_arr) == 0:
            print(f"  WARNING: no {poi_type} POIs, column set to NaN")
            df[col] = np.nan
            continue
        scaled = poi_arr.copy()
        scaled[:, 0] *= km_per_lat
        scaled[:, 1] *= km_per_lon
        tree = cKDTree(scaled)
        dists, _ = tree.query(pts_apt, k=1)
        df[col] = dists.astype(np.float32)
        print(f"    {col}: min={dists.min():.2f} km, median={np.median(dists):.2f} km, max={dists.max():.2f} km")

    return df


def main():
    # ── 1. Load and filter apartments ───────────────────────────────────────
    print("Loading merged_clean.csv...")
    merged = pd.read_csv(
        os.path.join(DATA_DIR, "processed", "merged_clean.csv"),
        usecols=["date", "price", "level", "levels", "rooms", "area",
                 "kitchen_area", "geo_lat", "geo_lon", "object_type",
                 "region", "source", "dist_to_center"],
        parse_dates=["date"],
    )
    city_codes = list(CITIES.keys())
    df_cities = merged[merged["region"].isin(city_codes)].copy()
    print(f"Apartments in 3 cities: {len(df_cities):,}")
    for c in city_codes:
        print(f"  region {c} ({CITIES[c]['name']}): {(df_cities['region']==c).sum():,}")

    # ── 2. Extract POIs and compute distances ────────────────────────────────
    all_frames = []
    for region_code, cfg in CITIES.items():
        print(f"\n=== {cfg['name']} (region {region_code}) ===")
        subset = df_cities[df_cities["region"] == region_code].copy()
        pois = extract_pois(region_code)
        lat_center = (cfg["bbox"][0] + cfg["bbox"][2]) / 2
        subset = compute_distances(subset, pois, lat_center)
        all_frames.append(subset)

    # ── 3. Save ──────────────────────────────────────────────────────────────
    result = pd.concat(all_frames, ignore_index=True)
    out_path = os.path.join(DATA_DIR, "processed", "cities_3_with_infra.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result):,} rows -> {out_path}")
    print(result[["dist_metro", "dist_school", "dist_hospital", "dist_park"]].describe())


if __name__ == "__main__":
    main()
