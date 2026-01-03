import geopandas as gpd
import rasterio
import rasterio.merge
import rasterio.mask
from rasterio.enums import Resampling
import fiona
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import glob
import os

# --------------------------------------------------
# Inputs
# --------------------------------------------------
GDB_PATH = "Nepal_Boundaries.gdb"
LAYER_NAME = "Nepal_International_boundary"
DEM_DIR = "dem_tiles"

# Downsampling factor (increase if memory is low)
DOWNSCALE_FACTOR = 6   # 6 → ~180 m resolution

# Plot aesthetics
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 140

# --------------------------------------------------
# Load Nepal boundary
# --------------------------------------------------
nepal = gpd.read_file(GDB_PATH, layer=LAYER_NAME)
nepal = nepal.to_crs(epsg=4326)

# --------------------------------------------------
# Read DEM tiles
# --------------------------------------------------
dem_files = sorted(glob.glob(os.path.join(DEM_DIR, "*_dem.tif")))
if not dem_files:
    raise RuntimeError("No DEM files found")

datasets = [rasterio.open(f) for f in dem_files]

# --------------------------------------------------
# Mosaic DEM
# --------------------------------------------------
mosaic, transform = rasterio.merge.merge(datasets)

meta = datasets[0].meta.copy()
meta.update({
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": transform,
    "dtype": "float32"
})

# --------------------------------------------------
# Clip to Nepal boundary
# --------------------------------------------------
with rasterio.io.MemoryFile() as memfile:
    with memfile.open(**meta) as tmp:
        tmp.write(mosaic.astype("float32"))
        clipped, clipped_transform = rasterio.mask.mask(
            tmp,
            nepal.geometry,
            crop=True,
            nodata=-9999
        )

dem = clipped[0]
dem[dem == -9999] = np.nan

# --------------------------------------------------
# Downsample for plotting
# --------------------------------------------------
new_height = dem.shape[0] // DOWNSCALE_FACTOR
new_width = dem.shape[1] // DOWNSCALE_FACTOR

dem_ds = np.empty((new_height, new_width), dtype=np.float32)

with rasterio.io.MemoryFile() as memfile:
    profile = meta.copy()
    profile.update({
        "height": dem.shape[0],
        "width": dem.shape[1],
        "transform": clipped_transform
    })
    with memfile.open(**profile) as src:
        src.write(dem, 1)
        dem_ds = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.average
        )
        ds_transform = src.transform * src.transform.scale(
            dem.shape[1] / new_width,
            dem.shape[0] / new_height
        )

# --------------------------------------------------
# Plot
# --------------------------------------------------
extent = [
    ds_transform[2],
    ds_transform[2] + ds_transform[0] * dem_ds.shape[1],
    ds_transform[5] + ds_transform[4] * dem_ds.shape[0],
    ds_transform[5],
]

# Multi-tone colormap with clear elevation breaks
zone_bounds = [0, 300, 800, 2000, 3000, 4000, 5000, 7000, 9000]
zone_colors = [
    "#f7fcf5",  # low plains
    "#d9f0d3",
    "#addd8e",
    "#78c679",
    "#41ab5d",
    "#238443",
    "#1a9850",
    "#225ea8",
]
cmap = ListedColormap(zone_colors)
cmap.set_bad("none", alpha=0)
norm = BoundaryNorm(zone_bounds, cmap.N)

fig, ax = plt.subplots(figsize=(11, 9))

img = ax.imshow(
    dem_ds,
    extent=extent,
    cmap=cmap,
    norm=norm,
    interpolation="nearest",
)

nepal.boundary.plot(ax=ax, color="black", linewidth=1.1)

cbar_ticks = [150, 1500, 5500]
cbar_ticklabels = [
    "Tarai (60-300 m)\nFlat alluvial plains",
    "Pahad (300-3,000 m)\nMid-hills and valleys",
    "Himal (>3,000 m)\nHigh peaks and glaciers",
]
cbar = plt.colorbar(
    img,
    ax=ax,
    shrink=0.74,
    pad=0.02,
    ticks=cbar_ticks,
)
cbar.ax.set_yticklabels(cbar_ticklabels)
cbar.set_label("Elevation zones of Nepal", labelpad=12)

ax.set_title("Topographical Map of Nepal Using (ASTER GDEM v3)", pad=12)
ax.set_xlabel("Longitude", labelpad=8)
ax.set_ylabel("Latitude", labelpad=8)
ax.tick_params(axis="both", which="major", labelsize=9)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

# North arrow
ax.annotate(
    "N",
    xy=(0.93, 0.88),
    xytext=(0.93, 0.68),
    xycoords="axes fraction",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, shrinkA=0, shrinkB=2),
)

plt.tight_layout()
plt.show()
