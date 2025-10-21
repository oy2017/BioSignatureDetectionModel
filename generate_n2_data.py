from multirex import Atmosphere, Planet, Star, System, Physics
import numpy as np
import pandas as pd

# This script generates a multiverse of planetary systems with N2 as the fill gas.

# --- User-defined settings ---
N_UNIVERSES = 3000
N_OBSERVATIONS = 1
SNR = 15

# --- Define Wavelength Grid ---
wl_min = 0.5
wl_max = 7.8
resolution = 200
wn_grid = Physics.wavenumber_grid(wl_min, wl_max, resolution)

# --- Stratified Generation of a Balanced Dataset ---
print("Generating a balanced N2 dataset with 5000 universes...")

bio_threshold_ch4 = -6
bio_threshold_o3 = -7
margin = 0.0

composition_profiles = {
    "biosignature": {
        "H2O": (-10, -1), "CO": (-9, -3), "CO2": (-9, -3), "NH3": (-9, -3),
        "CH4": (bio_threshold_ch4 + margin, -3),
        "O3": (bio_threshold_o3 + margin, -1),
    },
    "nonbio_ch4": {
        "H2O": (-10, -1), "CO": (-9, -3), "CO2": (-9, -3), "NH3": (-9, -3),
        "CH4": (bio_threshold_ch4 + margin, -3),
        "O3": (-10, bio_threshold_o3 - margin),
    },
    "nonbio_o3": {
        "H2O": (-10, -1), "CO": (-9, -3), "CO2": (-9, -3), "NH3": (-9, -3),
        "CH4": (-9, bio_threshold_ch4 - margin),
        "O3": (bio_threshold_o3 + margin, -1),
    },
    "nonbio_none": {
        "H2O": (-10, -1), "CO": (-9, -3), "CO2": (-9, -3), "NH3": (-9, -3),
        "CH4": (-9, bio_threshold_ch4 - margin),
        "O3": (-10, bio_threshold_o3 - margin),
    }
}

generation_plan = [
    {"name": "biosignature", "count": 1500},
    {"name": "nonbio_ch4", "count": 500},
    {"name": "nonbio_o3", "count": 500},
    {"name": "nonbio_none", "count": 500},
]

all_spectra = []
star = Star(
    temperature=(2500, 7500),
    radius=(0.1, 1.7),
    mass=(0.1, 1.7)
)

for item in generation_plan:
    profile_name = item["name"]
    profile_composition = composition_profiles[profile_name]
    count = item["count"]
    
    print(f"--- Generating {count} N2 planets for profile: {profile_name} ---")
    
    atmosphere = Atmosphere(
        temperature=(500, 2500),
        base_pressure=(1e5, 10e5),
        top_pressure=(1, 10),
        composition=profile_composition,
        fill_gas='N2'
    )
    
    planet = Planet(
        radius=(1.0, 15.0),
        mass=(1.0, 500.0),
        atmosphere=atmosphere
    )
    
    system = System(
        planet=planet,
        star=star,
        sma=(0.01, 0.5)
    )
    
    system.make_tm()
    
    results = system.explore_multiverse(
        wn_grid=wn_grid,
        n_universes=count,
        n_observations=N_OBSERVATIONS,
        snr=SNR,
        header=True,
        path=None,
        n_jobs=-1
    )
    
    all_spectra.append(results["spectra"])

print("\n--- Combining N2 datasets ---")
spectra_df = pd.concat(all_spectra, ignore_index=True)

def check_biosignature(row):
    if (row.get('atm CH4', -99) >= bio_threshold_ch4 and
        row.get('atm O3', -99) >= bio_threshold_o3):
        return 'yes'
    else:
        return 'no'

spectra_df['biosignature'] = spectra_df.apply(check_biosignature, axis=1)
spectra_df.to_parquet("multirex_spectra_n2.parquet")

print("N2 dataset created and saved to multirex_spectra_n2.parquet")
