import argparse
from multirex import Atmosphere, Planet, Star, System, Physics
import numpy as np
import pandas as pd

# This script generates a multiverse of planetary systems with a user-defined fill gas.

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
# Add argument parser
parser = argparse.ArgumentParser(description="Generate a multiverse dataset with a specified fill gas.")
parser.add_argument("fill_gas", type=str, help="The fill gas for the atmosphere (e.g., H2, N2).")
args = parser.parse_args()

fill_gas = args.fill_gas.upper() # Convert to uppercase for consistency

print(f"Generating a balanced {fill_gas} dataset with 5000 universes...")

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

# --- Define planet parameters based on fill_gas ---
if fill_gas == 'H2':
    planet_radius_range = (5.0, 15.0)
    planet_mass_range = (20.0, 300.0)
    # These ranges are chosen to represent physically plausible H2-dominated atmospheres
    # for super-Earths and mini-Neptunes, avoiding parameter space where H2 atmospheres
    # might be unstable or unrealistic for the given stellar parameters.
elif fill_gas == 'N2':
    planet_radius_range = (1.0, 15.0)
    planet_mass_range = (1.0, 500.0)
    # These ranges are chosen to represent physically plausible N2-dominated atmospheres
    # for rocky planets and super-Earths, allowing for a broader range of sizes and masses
    # compared to H2-dominated planets.
else:
    # Default or raise an error for unsupported fill_gas
    print(f"Warning: Unsupported fill_gas '{fill_gas}'. Using default N2 planet parameters.")
    planet_radius_range = (1.0, 15.0)
    planet_mass_range = (1.0, 500.0)


for item in generation_plan:
    profile_name = item["name"]
    profile_composition = composition_profiles[profile_name]
    count = item["count"]
    
    print(f"--- Generating {count} {fill_gas} planets for profile: {profile_name} ---")
    
    atmosphere = Atmosphere(
        temperature=(500, 2500),
        base_pressure=(1e5, 10e5),
        top_pressure=(1, 10),
        composition=profile_composition,
        fill_gas=fill_gas # Use the input fill_gas
    )
    
    planet = Planet(
        radius=planet_radius_range, # Use dynamic radius
        mass=planet_mass_range,     # Use dynamic mass
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

print(f"\n--- Combining {fill_gas} datasets ---")
spectra_df = pd.concat(all_spectra, ignore_index=True)

def check_biosignature(row):
    if (row.get('atm CH4', -99) >= bio_threshold_ch4 and
        row.get('atm O3', -99) >= bio_threshold_o3):
        return 'yes'
    else:
        return 'no'

spectra_df['biosignature'] = spectra_df.apply(check_biosignature, axis=1)
output_filename = f"multirex_spectra_{fill_gas}.parquet" # Dynamic filename
spectra_df.to_parquet(output_filename)

print(f"{fill_gas} dataset created and saved to {output_filename}")