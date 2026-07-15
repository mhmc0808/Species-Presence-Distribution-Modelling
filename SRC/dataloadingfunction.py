#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import reverse_geocoder as rg
from scipy.spatial import cKDTree
from netCDF4 import Dataset, num2date
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


# function for creating path to data
def concatlink(name): 
    return folder + "/" + name

# function to load in the data and refine to only US

def load_full_dataset(df, folder):


    coords = list(zip(df.latitude, df.longitude))
    results = rg.search(coords)
    df["country_code"] = [r["cc"] for r in results]
    df["city"] = [r["name"] for r in results]
    df["state"] = [r["admin1"] for r in results]
    df["county"] = [r["admin2"] for r in results]

    # Keep only continental US add washington dc 
    continental = {
        "Alabama","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","Florida",
        "Georgia","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine",
        "Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
        "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina",
        "North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina",
        "South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington",
        "West Virginia","Wisconsin","Wyoming", 'Washington, D.C.'
    }
    df = df[(df.country_code == "US") & (df.state.isin(continental))].drop(columns="country_code")
    
   # function for adding features over to data, specifically the temperature, precipitation, soil moisture - these features we use mean and var
    def attach_nc_feature(df, path, var, prefix):
    
        ds = Dataset(path)
        lat, lon = ds["lat"][:], ds["lon"][:]
        data = ds[var][:]
        times = num2date(ds["time"][:], units=ds["time"].units, calendar=getattr(ds["time"], "calendar", "gregorian"))
        mask = np.array([t.year == 2020 for t in times])
        data = data[mask, ...]
        avg = np.ma.mean(data, axis=0).filled(np.nan)
        varr = np.ma.var(data, axis=0).filled(np.nan)
        latg, long = np.meshgrid(lat, lon, indexing="ij")
        grid = pd.DataFrame({
            "latitude": latg.ravel(),
            "longitude": long.ravel(),
            f"{prefix}_mean": avg.ravel(),
            f"{prefix}_var": varr.ravel()
        })
        # use cKDTree to add nearest feature data points for lat and lon 
        tree = cKDTree(grid[["latitude", "longitude"]])
        _, idx = tree.query(df[["latitude", "longitude"]])
        df[f"{prefix}_mean"] = grid[f"{prefix}_mean"].values[idx]
        df[f"{prefix}_var"] = grid[f"{prefix}_var"].values[idx]
        return df

    #  Use functions above to add over features to data
    
    # add climate features
    df = attach_nc_feature(df, concatlink("nclimgrid_tavg.nc"), "tavg", "temp")
    df = attach_nc_feature(df, concatlink("nclimgrid_prcp.nc"), "prcp", "prcp")

    # Wind speed
   
    uwnd, vwnd = Dataset(concatlink("uwnd.10m.gauss.2020.nc")), Dataset(concatlink("vwnd.10m.gauss.2020.nc"))
    lat, lon = uwnd["lat"][:], uwnd["lon"][:]
    
    # Compute total wind speed magnitude and average over time
    ws = np.sqrt(uwnd["uwnd"][:, 0, :, :]**2 + vwnd["vwnd"][:, 0, :, :]**2)
    avg = np.ma.mean(ws, axis=0).filled(np.nan)
    latg, long = np.meshgrid(lat, lon, indexing="ij")
    grid = pd.DataFrame({"latitude": latg.ravel(), "longitude": long.ravel(), "avg_wind": avg.ravel()})
    tree = cKDTree(grid[["latitude", "longitude"]])
    _, idx = tree.query(df[["latitude", "longitude"]])
    df["avg_wind"] = grid["avg_wind"].values[idx]

    # Soil moisture

    df = attach_nc_feature(df, concatlink("soilw.0-10cm.gauss.2020.nc"), "soilw", "soil")

    # NDVI

    nd = Dataset(concatlink("VIIRS-Land_v001_NPP13C1_S-NPP_20200101_c20240126161431.nc"))
    lat, lon = nd["latitude"][:], nd["longitude"][:]
    ndvi = nd["NDVI"][:]
    times = num2date(nd["time"][:], units=nd["time"].units, calendar=getattr(nd["time"], "calendar", "gregorian"))
    mask = np.array([t.year == 2020 for t in times])
    avg = np.ma.mean(ndvi[mask, :, :], axis=0).filled(np.nan)
    latg, long = np.meshgrid(lat, lon, indexing="ij")
    grid = pd.DataFrame({"latitude": latg.ravel(), "longitude": long.ravel(), "avg_ndvi": avg.ravel()})
    tree = cKDTree(grid[["latitude", "longitude"]])
    _, idx = tree.query(df[["latitude", "longitude"]])
    df["avg_ndvi"] = grid["avg_ndvi"].values[idx]
    

    # Distance to coastline
    
    coast = gpd.read_file("https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip")
    coast = coast.cx[-130:-60, 15:60]
    coast = coast.to_crs("EPSG:3857")
    coast_union = coast.unary_union
    pts = gpd.GeoDataFrame(
    geometry=[Point(xy) for xy in zip(df.longitude, df.latitude)],
    crs="EPSG:4326"
    ).to_crs("EPSG:3857")
    pts["dist_coast"] = pts.geometry.apply(lambda p: p.distance(coast_union))
    df["distance_to_coast"] = pts["dist_coast"].values

    # Proportion of area that is farmland
    farmland = pd.read_csv(concatlink("farmland_proportion_by_county.csv"))[["State", "County", "Value"]]
    
    farmland = farmland.rename(columns={"Value": "Farmland Value", "State" : "state", "County": "county"})
    farmland['Farmland Value'] = pd.to_numeric(farmland['Farmland Value'], errors = 'coerce' )
    
    df["county"] = df["county"].str.replace(r"\s+County$", "", regex=True).str.upper()
    df["state"] = df["state"].str.upper()
    farmland["state"] = farmland["state"].str.upper()
    
    
    df = df.merge(farmland, on=["state", "county"], how="left")
    
    #### deal with nan values ###
    
    for col in df.select_dtypes(include=["float"]).columns:
        
            
         if col in ["latitude", "longitude"]:
             
             continue
        
         valid_mask = ~np.isnan(df[col])
        
        
         valid_coords = np.column_stack((df.loc[valid_mask, "latitude"], df.loc[valid_mask, "longitude"]))
         valid_values = df.loc[valid_mask, col].values
         valid_tree = cKDTree(valid_coords)
        
         nan_mask = np.isnan(df[col])
         if nan_mask.any():
             query_coords_nan = np.column_stack((df.loc[nan_mask, "latitude"], df.loc[nan_mask, "longitude"]))
             _, nearest_idx = valid_tree.query(query_coords_nan)
             df.loc[nan_mask, col] = valid_values[nearest_idx]
    
         # Fill any remaining nans with column mean
         df[col].fillna(np.nanmean(valid_values), inplace=True)
    
    return df

folder = "/Users/michael/Desktop/AML Repo/AML-Group-work-/AML Data"

d = np.load(folder + "/" + "species_train.npz", allow_pickle=True)

df = pd.DataFrame(d["train_locs"], columns=["latitude", "longitude"])
df["species_id"] = d["train_ids"]
species_names = dict(zip(d["taxon_ids"], d["taxon_names"]))
df["species_name"] = df["species_id"].map(species_names)

df = load_full_dataset(df, "/Users/michael/Desktop/AML Repo/AML-Group-work-/AML Data")






### ---- Little bit of EDA ----

eda_feats = ['species_id', 'State', 'distance_to_coast', 'avg_ndvi','temp_mean', 
             'temp_var', 'prcp_mean', 'prcp_var', 'avg_wind', 
             'soil_mean', 'Farmland Value']

# variable name changes for nice EDA plot
data_eda = df[eda_feats]
data_eda = data_eda.rename(columns={"avg_wind": "avg_windspeed"})
data_eda = data_eda.rename(columns={"soil_mean": "avg_soil_moisture"})
data_eda = data_eda.rename(columns={"avg_ndvi": "NDVI"})
data_eda = data_eda.rename(columns={"Farmland Value": "proportion_farmland"})
data_eda = data_eda.drop(columns=['species_id', 'State'])

# plot feature correlation heatmap
corr = data_eda.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=True, fmt=".2f")
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(rotation=0, fontsize=14)   # keep y-labels horizontal (usually best)
plt.title("")
plt.show()


