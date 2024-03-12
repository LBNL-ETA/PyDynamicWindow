"""
*** Copyright Notice ***

PyDynamicWindow Copyright (c) 2024, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.
"""

import glob
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from scipy.ndimage import zoom

# Before you run this file:
# download zip files for minimum temperature (°C),
# maximum temperature (°C), and average temperature (°C),
# in 10 minutes resolution from the website:
# https://www.worldclim.org/data/worldclim21.html
# and unzip the folders (wc2.1_10m_tmin, wc2.1_10m_tmax, and wc2.1_10m_tavg)
# to the folder 'Map_tiff_npy_files'


# Define paths
path = 'Map_tiff_npy_files'
folder_Tavg = 'wc2.1_10m_tavg'
folder_Tmax = 'wc2.1_10m_tmax'
folder_Tmin = 'wc2.1_10m_tmin'
file_tif = '*.tif'
file_GHI = 'world_input_GHI.tif'
# Function to read geotiff files


def read_geotiff(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)
        data[data == src.nodata] = np.nan
        return data, src.transform, src.crs

# Function to resize geotiff data
def resize_geotiff(data, scale_factor, resampling_method=Resampling.nearest):
    return zoom(data, scale_factor, order=resampling_method.value)

# Process GHI
A1, R1, crs1 = read_geotiff(f'{path}/{file_GHI}')

# Read Tavg, Tmax, Tmin, and GHI data
Tavg_files = glob.glob(f'{path}/{folder_Tavg}/{file_tif}')
Tmax_files = glob.glob(f'{path}/{folder_Tmax}/{file_tif}')
Tmin_files = glob.glob(f'{path}/{folder_Tmin}/{file_tif}')

# Process Tavg
Tavg_12m = np.zeros((1080*2160, 12))
for idx, file_name in enumerate(Tavg_files):
    A, _, _ = read_geotiff(file_name)
    Tavg_12m[:, idx] = A.flatten()
v_Tavg = np.nanmean(Tavg_12m, axis=1)
Tavg = v_Tavg.reshape(1080, 2160)

# Process Tmax
Tmax_12m = np.zeros((1080*2160, 12))
for idx, file_name in enumerate(Tmax_files):
    A, _, _ = read_geotiff(file_name)
    Tmax_12m[:, idx] = A.flatten()
v_Tmax = np.nanmax(Tmax_12m, axis=1)
Tmax = v_Tmax.reshape(1080, 2160)

# Process Tmin
Tmin_12m = np.zeros((1080*2160, 12))
for idx, file_name in enumerate(Tmin_files):
    A, _, _ = read_geotiff(file_name)
    Tmin_12m[:, idx] = A.flatten()
v_Tmin = np.nanmin(Tmin_12m, axis=1)
Tmin = v_Tmin.reshape(1080, 2160)

# Crop to 690x2160
lat_s = np.arange(181, 871)[:, None]
Tavg_s = Tavg[180:870, :]
Tmax_s = Tmax[180:870, :]
Tmin_s = Tmin[180:870, :]


filename_Tavg = f'{path}/world_input_Tavg.tif'
with rasterio.open(filename_Tavg, 'w', driver='GTiff', height=Tavg_s.shape[0], width=Tavg_s.shape[1], count=1, dtype='float32', crs=crs1, transform=R1) as dst:
    dst.write(Tavg_s, 1)

filename_Tmin = f'{path}/world_input_Tmin.tif'
with rasterio.open(filename_Tmin, 'w', driver='GTiff', height=Tmin_s.shape[0], width=Tmin_s.shape[1], count=1, dtype='float32', crs=crs1, transform=R1) as dst:
    dst.write(Tmin_s, 1)

filename_Tmax = f'{path}/world_input_Tmax.tif'
with rasterio.open(filename_Tmax, 'w', driver='GTiff', height=Tmax_s.shape[0], width=Tmax_s.shape[1], count=1, dtype='float32', crs=crs1, transform=R1) as dst:
    dst.write(Tmax_s, 1)