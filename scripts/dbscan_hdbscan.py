import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import StandardScaler

# Geographical tools
import geopandas as gpd    # DataFrame-like structure for geographic data with geometry columns
import contextily as ctx   # Fetches and overlays map tiles (basemap) on matplotlib axes
from shapely.geometry import Point

import warnings
warnings.filterwarnings('ignore')

import requests
import zipfile
import io
import os

# Download a ZIP file containing a basemap raster (TIFF) of Canada from IBM Cloud Object Storage
zip_file_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YcUk-ytgrPkmvZAh5bf7zA/Canada.zip'

# Extract to the current working directory
output_dir = './'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Download the ZIP file into memory (no temp file on disk)
response = requests.get(zip_file_url)
response.raise_for_status()  # Raise an error if the download failed

# Step 2: Open the in-memory ZIP and extract only TIFF files
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    for file_name in zip_ref.namelist():
        if file_name.endswith('.tif'):
            zip_ref.extract(file_name, output_dir)
            print(f"Downloaded and extracted: {file_name}")


def plot_clustered_locations(df, title='Museums Clustered by Proximity'):
    """
    Plots clustered locations and overlays on a basemap.

    Parameters:
    - df: DataFrame containing 'Latitude', 'Longitude', and 'Cluster' columns
    - title: str, title of the plot
    """

    # Create a GeoDataFrame: convert lat/lon columns to Shapely Point geometry
    # CRS EPSG:4326 is standard WGS84 geographic coordinates (degrees)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")

    # Reproject from geographic (degrees) to Web Mercator (metres)
    # Basemap tiles are in Web Mercator, so the CRS must match
    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(15, 10))

    # DBSCAN/HDBSCAN labels noise points as -1; split them out for distinct styling
    non_noise = gdf[gdf['Cluster'] != -1]
    noise = gdf[gdf['Cluster'] == -1]

    # Plot noise points (unclustered museums) with a red-edged black marker
    noise.plot(ax=ax, color='k', markersize=30, ec='r', alpha=1, label='Noise')

    # Plot clustered museums, coloured by cluster ID using the tab10 palette
    non_noise.plot(ax=ax, column='Cluster', cmap='tab10', markersize=30, ec='k', legend=False, alpha=0.6)

    # Overlay the downloaded Canada TIFF as the background basemap
    ctx.add_basemap(ax, source='./Canada.tif', zoom=4)

    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


# Load the Canadian Open Data Cultural and Arts Facilities dataset
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'
df = pd.read_csv(url, encoding="ISO-8859-1")

# Show the breakdown of facility types in the dataset
print(df['ODCAF_Facility_Type'].value_counts())

# Keep only museum entries
df = df[df['ODCAF_Facility_Type'] == 'museum']

# Retain only the coordinate columns needed for clustering
df = df[['Latitude', 'Longitude']]

# Remove rows where coordinates are missing (stored as '..' in this dataset)
df = df[df.Latitude != '..']

# Cast coordinates from strings to floats for numerical operations
df[['Latitude', 'Longitude']] = df[['Latitude', 'Longitude']].astype('float')

# Scale latitude by 2 to give it slightly more weight in the distance metric,
# compensating for the geographic distortion at Canada's latitude
coords_scaled = df.copy()
coords_scaled["Latitude"] = 2 * coords_scaled["Latitude"]

# --- DBSCAN clustering ---
# eps: radius of the neighbourhood around each point (in the scaled coordinate space)
# min_samples: minimum number of points required to form a dense region (core point)
# Points that don't belong to any dense region are labelled as noise (-1)
min_samples = 3
eps = 1.0
metric = 'euclidean'

dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(coords_scaled)
df['Cluster'] = dbscan.fit_predict(coords_scaled)

# Show how many museums fall into each cluster (and how many are noise)
df['Cluster'].value_counts()

plot_clustered_locations(df, title='Museums Clustered by Proximity (DBSCAN)')

# --- HDBSCAN clustering ---
# HDBSCAN is a hierarchical extension of DBSCAN that automatically determines
# the density threshold, making it more robust to varying cluster densities
# min_cluster_size: smallest number of points that can form a cluster
# min_samples=None: let HDBSCAN choose the default (equal to min_cluster_size)
min_samples = None
min_cluster_size = 3
hdb = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, metric='euclidean')

df['Cluster'] = hdb.fit_predict(coords_scaled)
df['Cluster'].value_counts()
