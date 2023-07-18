# ANNs_TCwin_functions
# define 7 different functions for various inputs and outputs
# output files include npy, png, tif

import sklearn
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import os


# Function 01
# Output: Etc,
# Input: Tsol_dark, Tsol_clear, FileName

def world_map_Etc_TsolAsInput(Tsol_dark, Tsol_clear, FileName):
    # input solar transmittance in clear state and dark state
    target_values = np.array([Tsol_clear, Tsol_dark])
    # load scaler
    # Get the current script directory
    script_dir = os.path.dirname(__file__)
    # Construct absolute path to the file you want to load
    file_path_scaler = os.path.join(script_dir, 'ANN_joblib_files', '401_Etc_Map_2226_Random_TsolAsTrain_save_split_scaler_05.joblib')
    scaler = load(file_path_scaler)
    # load model
    file_path_model = os.path.join(script_dir, 'ANN_joblib_files',
                                    '401_Etc_Map_2226_Random_TsolAsTrain_save_split_05.joblib')

    final_model_Etc = load(file_path_model)

    # load map data
    data_path_map = 'Map_tiff_npy_files'
    filename_Tavg = 'world_input_Tavg.tif'
    filename_Tmin = 'world_input_Tmin.tif'
    filename_Tmax = 'world_input_Tmax.tif'

    file_full_Tavg = os.path.join(script_dir, data_path_map, filename_Tavg)
    file_full_Tmin = os.path.join(script_dir, data_path_map, filename_Tmin)
    file_full_Tmax = os.path.join(script_dir, data_path_map, filename_Tmax)

    # read tif files
    # Tavg
    src_Tavg = rasterio.open(file_full_Tavg)
    array_Tavg = src_Tavg.read(1)
    # Tmin
    src_Tmin = rasterio.open(file_full_Tmin)
    array_Tmin = src_Tmin.read(1)
    # Tmax
    src_Tmax = rasterio.open(file_full_Tmax)
    array_Tmax = src_Tmax.read(1)
    # lat
    height = array_Tavg.shape[0]
    width = array_Tavg.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src_Tavg.transform, rows, cols)
    array_lons = np.array(xs)
    array_lats = np.array(ys)

    file_full_GVI = os.path.join(script_dir, data_path_map, '204_GVI_MLP_2226_grid_save.py.npy')
    array_gvi = np.load(file_full_GVI)

    list_Tavg = array_Tavg.flatten()
    list_Tmin = array_Tmin.flatten()
    list_Tmax = array_Tmax.flatten()
    list_lats = array_lats.flatten()
    list_gvi = array_gvi.flatten()
    # add Low_Tsol and High_Tsol, B = np.full_like(A, 1)
    list_Low_Tsol = np.full_like(list_Tavg, target_values[1])
    list_High_Tsol = np.full_like(list_Tavg, target_values[0])

    nan_mask = np.isnan(list_gvi) | np.isnan(list_Tavg) | np.isnan(list_Tmin) | np.isnan(list_Tmax)
    list_lats[nan_mask] = np.nan

    data_map_input = pd.DataFrame({
        'abs_latitude': np.abs(list_lats),
        'gvi': list_gvi,
        'Tair_ave': list_Tavg,
        'Tair_min': list_Tmin,
        'Tair_max': list_Tmax,
        'Low_Tsol': list_Low_Tsol,
        'High_Tsol': list_High_Tsol,
    })

    list_Etc = np.full_like(list_lats, np.nan)

    # map heatmap En
    data_map_input_scale_Etc = scaler.transform(data_map_input)
    list_Etc[~nan_mask] = final_model_Etc.predict(data_map_input_scale_Etc[~nan_mask])
    reshaped_array = list_Etc.reshape((height, width))

    # Save as a NumPy npy file
    np.save(f'{FileName}_Etc_TsolAsInput.npy', reshaped_array)
    # Draw the wide heatmap occupying the entire subplot
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(reshaped_array, cmap='viridis', extent=[-180, 180, -55, 60], aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.grid(True, linestyle='dashed', linewidth=0.5, color='lightgrey')
    ax.set_xlabel('Longitude (\u00B0)')
    ax.set_ylabel('Latitude (\u00B0)')
    ax.set_title(
        f'$\u0394E_{{tc}}$ (kWh/m$^2$) by ANN, when $\u03C4_{{dark}}$ = {target_values[1]:.2f} and $\u03C4_{{clear}}$ = {target_values[0]:.2f}')
    plt.tight_layout()
    # Save the figure as a PNG file
    plt.savefig(f'{FileName}_Etc_TsolAsInput.png', dpi=300)
    # Show the figure
    # plt.show()

    # Save as a GeoTIFF file
    with rasterio.open(
            f'{FileName}_Etc_TsolAsInput.tif',
            'w',
            driver='GTiff',
            height=reshaped_array.shape[0],
            width=reshaped_array.shape[1],
            count=1,
            dtype=reshaped_array.dtype,
            crs=src_Tavg.crs,
            transform=src_Tavg.transform,
    ) as dst_Etc:
        dst_Etc.write(reshaped_array, 1)

# Function 02
# Output: En,
# Input: Tsol_dark, Tsol_clear, FileName
# for two inputs function, En should be a non-negative value because of opt Ttran

def world_map_En_TsolAsInput(Tsol_dark, Tsol_clear, FileName):
    # input solar transmittance in clear state and dark state
    target_values = np.array([Tsol_clear, Tsol_dark])
    # load scaler
    # Get the current script directory
    script_dir = os.path.dirname(__file__)
    # Construct absolute path to the file you want to load
    file_path_scaler = os.path.join(script_dir, 'ANN_joblib_files', '402_En_Map_2226_Random_TsolAsTrain_save_scaler.joblib')
    scaler = load(file_path_scaler)
    # load model
    file_path_model = os.path.join(script_dir, 'ANN_joblib_files',
                                    '402_En_Map_2226_Random_TsolAsTrain_save_02.joblib')
    final_model_En = load(file_path_model)

    # load map data
    data_path_map = 'Map_tiff_npy_files'
    filename_Tavg = 'world_input_Tavg.tif'
    filename_Tmin = 'world_input_Tmin.tif'
    filename_Tmax = 'world_input_Tmax.tif'

    file_full_Tavg = os.path.join(script_dir, data_path_map, filename_Tavg)
    file_full_Tmin = os.path.join(script_dir, data_path_map, filename_Tmin)
    file_full_Tmax = os.path.join(script_dir, data_path_map, filename_Tmax)

    # read tif files
    # Tavg
    src_Tavg = rasterio.open(file_full_Tavg)
    array_Tavg = src_Tavg.read(1)
    # Tmin
    src_Tmin = rasterio.open(file_full_Tmin)
    array_Tmin = src_Tmin.read(1)
    # Tmax
    src_Tmax = rasterio.open(file_full_Tmax)
    array_Tmax = src_Tmax.read(1)
    # lat
    height = array_Tavg.shape[0]
    width = array_Tavg.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src_Tavg.transform, rows, cols)
    array_lons = np.array(xs)
    array_lats = np.array(ys)

    file_full_GVI = os.path.join(script_dir, data_path_map, '204_GVI_MLP_2226_grid_save.py.npy')
    array_gvi = np.load(file_full_GVI)

    list_Tavg = array_Tavg.flatten()
    list_Tmin = array_Tmin.flatten()
    list_Tmax = array_Tmax.flatten()
    list_lats = array_lats.flatten()
    list_gvi = array_gvi.flatten()
    # add Low_Tsol and High_Tsol, B = np.full_like(A, 1)
    list_Low_Tsol = np.full_like(list_Tavg, target_values[1])
    list_High_Tsol = np.full_like(list_Tavg, target_values[0])

    nan_mask = np.isnan(list_gvi) | np.isnan(list_Tavg) | np.isnan(list_Tmin) | np.isnan(list_Tmax)
    list_lats[nan_mask] = np.nan

    data_map_input = pd.DataFrame({
        'abs_latitude': np.abs(list_lats),
        'gvi': list_gvi,
        'Tair_ave': list_Tavg,
        'Tair_min': list_Tmin,
        'Tair_max': list_Tmax,
        'Low_Tsol': list_Low_Tsol,
        'High_Tsol': list_High_Tsol,
    })

    list_En = np.full_like(list_lats, np.nan)

    # map heatmap En
    data_map_input_scale_En = scaler.transform(data_map_input)
    list_En[~nan_mask] = final_model_En.predict(data_map_input_scale_En[~nan_mask])
    # non-negative value
    list_En[list_En < 0] = 0
    reshaped_array = list_En.reshape((height, width))

    # Save as a NumPy npy file
    np.save(f'{FileName}_En_TsolAsInput.npy', reshaped_array)
    # Draw the wide heatmap occupying the entire subplot
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(reshaped_array, cmap='viridis', extent=[-180, 180, -55, 60], aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.grid(True, linestyle='dashed', linewidth=0.5, color='lightgrey')
    ax.set_xlabel('Longitude (\u00B0)')
    ax.set_ylabel('Latitude (\u00B0)')
    ax.set_title(
        f'$\u0394E_{{n}}$ (kWh/m$^2$) by ANN, when $\u03C4_{{dark}}$ = {target_values[1]:.2f} and $\u03C4_{{clear}}$ = {target_values[0]:.2f}')
    plt.tight_layout()
    # Save the figure as a PNG file
    plt.savefig(f'{FileName}_En_TsolAsInput.png', dpi=300)
    # Show the figure
    # plt.show()

    # Save as a GeoTIFF file
    with rasterio.open(
            f'{FileName}_En_TsolAsInput.tif',
            'w',
            driver='GTiff',
            height=reshaped_array.shape[0],
            width=reshaped_array.shape[1],
            count=1,
            dtype=reshaped_array.dtype,
            crs=src_Tavg.crs,
            transform=src_Tavg.transform,
    ) as dst_En:
        dst_En.write(reshaped_array, 1)

# Function 03
# Output: Ttran,
# Input: Tsol_dark, Tsol_clear, FileName


def world_map_Ttran_TsolAsInput(Tsol_dark, Tsol_clear, FileName):
    # input solar transmittance in clear state and dark state
    target_values = np.array([Tsol_clear, Tsol_dark])
    # load scaler
    # Get the current script directory
    script_dir = os.path.dirname(__file__)
    # Construct absolute path to the file you want to load
    file_path_scaler = os.path.join(script_dir, 'ANN_joblib_files',
                                    '403_optTtran_Map_2226_Random_TsolAsTrain_save_scaler.joblib')
    scaler = load(file_path_scaler)
    # load model
    file_path_model = os.path.join(script_dir, 'ANN_joblib_files',
                                   '403_optTtran_Map_2226_Random_TsolAsTrain_save_03.joblib')
    final_model_Ttran = load(file_path_model)

    # load map data
    data_path_map = 'Map_tiff_npy_files'
    filename_Tavg = 'world_input_Tavg.tif'
    filename_Tmin = 'world_input_Tmin.tif'
    filename_Tmax = 'world_input_Tmax.tif'

    file_full_Tavg = os.path.join(script_dir, data_path_map, filename_Tavg)
    file_full_Tmin = os.path.join(script_dir, data_path_map, filename_Tmin)
    file_full_Tmax = os.path.join(script_dir, data_path_map, filename_Tmax)

    # read tif files
    # Tavg
    src_Tavg = rasterio.open(file_full_Tavg)
    array_Tavg = src_Tavg.read(1)
    # Tmin
    src_Tmin = rasterio.open(file_full_Tmin)
    array_Tmin = src_Tmin.read(1)
    # Tmax
    src_Tmax = rasterio.open(file_full_Tmax)
    array_Tmax = src_Tmax.read(1)
    # lat
    height = array_Tavg.shape[0]
    width = array_Tavg.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src_Tavg.transform, rows, cols)
    array_lons = np.array(xs)
    array_lats = np.array(ys)

    file_full_GVI = os.path.join(script_dir, data_path_map, '204_GVI_MLP_2226_grid_save.py.npy')
    array_gvi = np.load(file_full_GVI)

    list_Tavg = array_Tavg.flatten()
    list_Tmin = array_Tmin.flatten()
    list_Tmax = array_Tmax.flatten()
    list_lats = array_lats.flatten()
    list_gvi = array_gvi.flatten()
    # add Low_Tsol and High_Tsol, B = np.full_like(A, 1)
    list_Low_Tsol = np.full_like(list_Tavg, target_values[1])
    list_High_Tsol = np.full_like(list_Tavg, target_values[0])

    nan_mask = np.isnan(list_gvi) | np.isnan(list_Tavg) | np.isnan(list_Tmin) | np.isnan(list_Tmax)
    list_lats[nan_mask] = np.nan

    data_map_input = pd.DataFrame({
        'abs_latitude': np.abs(list_lats),
        'gvi': list_gvi,
        'Tair_ave': list_Tavg,
        'Tair_min': list_Tmin,
        'Tair_max': list_Tmax,
        'Low_Tsol': list_Low_Tsol,
        'High_Tsol': list_High_Tsol,
    })

    list_Ttran = np.full_like(list_lats, np.nan)

    # map heatmap En
    data_map_input_scale_Ttran = scaler.transform(data_map_input)
    list_Ttran[~nan_mask] = final_model_Ttran.predict(data_map_input_scale_Ttran[~nan_mask])
    reshaped_array = list_Ttran.reshape((height, width))

    # Save as a NumPy npy file
    np.save(f'{FileName}_Ttran_TsolAsInput.npy', reshaped_array)
    # Draw the wide heatmap occupying the entire subplot
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(reshaped_array, cmap='viridis', extent=[-180, 180, -55, 60], aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.grid(True, linestyle='dashed', linewidth=0.5, color='lightgrey')
    ax.set_xlabel('Longitude (\u00B0)')
    ax.set_ylabel('Latitude (\u00B0)')
    ax.set_title(
        f'Opt $T_{{tran}}$ (\u00B0C) by ANN, when $\u03C4_{{dark}}$ = {target_values[1]:.2f} and $\u03C4_{{clear}}$ = {target_values[0]:.2f}')
    plt.tight_layout()
    # Save the figure as a PNG file
    plt.savefig(f'{FileName}_Ttran_TsolAsInput.png', dpi=300)
    # Show the figure
    # plt.show()

    # Save as a GeoTIFF file
    with rasterio.open(
            f'{FileName}_Ttran_TsolAsInput.tif',
            'w',
            driver='GTiff',
            height=reshaped_array.shape[0],
            width=reshaped_array.shape[1],
            count=1,
            dtype=reshaped_array.dtype,
            crs=src_Tavg.crs,
            transform=src_Tavg.transform,
    ) as dst_Ttran:
        dst_Ttran.write(reshaped_array, 1)

# Function 04
# Output: TCRI,
# Input: Tsol_dark, Tsol_clear, Etc_npy_array, En_npy_array, FileName

def world_map_TCRI_TsolAsInput(Tsol_dark, Tsol_clear, Etc_npy_array, En_npy_array, FileName):
    # input solar transmittance in clear state and dark state
    target_values = np.array([Tsol_clear, Tsol_dark])
    # Get the current script directory
    script_dir = os.path.dirname(__file__)
    # load Etc and En
    array_Etc = np.load(Etc_npy_array)
    array_En = np.load(En_npy_array)
    # array to list
    list_Etc = array_Etc.flatten()
    list_En = array_En.flatten()
    # ReLU
    list_Etc[list_Etc < 0] = 0
    list_En[list_En < 0] = 0

    # load Tavg
    data_path_map = 'Map_tiff_npy_files'
    filename_Tavg = 'world_input_Tavg.tif'
    file_full_Tavg = os.path.join(script_dir, data_path_map, filename_Tavg)
    # read tif files
    # Tavg
    src_Tavg = rasterio.open(file_full_Tavg)

    # define empty list for output Etc
    list_TCRI = list()
    # loop
    for value_Etc, value_En in zip(list_Etc, list_En):
        if (np.isnan(value_Etc) or np.isnan(value_En)):
            list_TCRI.append(np.nan)
        else:
            temp_input = value_Etc*value_En
            list_TCRI.append(temp_input)

    # reshape list
    reshaped_array = np.reshape(list_TCRI, (690, 2160))

    # Save as a NumPy npy file
    np.save(f'{FileName}_TCRI_TsolAsInput.npy', reshaped_array)
    # Draw the wide heatmap occupying the entire subplot
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(reshaped_array, cmap='viridis', extent=[-180, 180, -55, 60], aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.grid(True, linestyle='dashed', linewidth=0.5, color='lightgrey')
    ax.set_xlabel('Longitude (\u00B0)')
    ax.set_ylabel('Latitude (\u00B0)')
    ax.set_title(
        f'TCRI (kWh$^2$/m$^4$) by ANN, when $\u03C4_{{dark}}$ = {target_values[1]:.2f} and $\u03C4_{{clear}}$ = {target_values[0]:.2f}')
    plt.tight_layout()
    # Save the figure as a PNG file
    plt.savefig(f'{FileName}_TCRI_TsolAsInput.png', dpi=300)
    # Show the figure
    # plt.show()

    # Save as a GeoTIFF file
    with rasterio.open(
            f'{FileName}_TCRI_TsolAsInput.tif',
            'w',
            driver='GTiff',
            height=reshaped_array.shape[0],
            width=reshaped_array.shape[1],
            count=1,
            dtype=reshaped_array.dtype,
            crs=src_Tavg.crs,
            transform=src_Tavg.transform,
    ) as dst_TCRI:
        dst_TCRI.write(reshaped_array, 1)

# Function 05
# Output: Etc,
# Input: Tsol_dark, Tsol_clear, Ttran, FileName

def world_map_Etc_TsolTtranAsInput(Tsol_dark, Tsol_clear, Ttran, FileName):
    # input solar transmittance in clear state and dark state, and transition temperature
    target_values = np.array([Tsol_clear, Tsol_dark, Ttran])
    # load scaler
    # Get the current script directory
    script_dir = os.path.dirname(__file__)
    # Construct absolute path to the file you want to load
    file_path_scaler = os.path.join(script_dir, 'ANN_joblib_files',
                                    '501_Etc_Map_2226_Random_AlllAsTrain_save_scaler.joblib')
    scaler = load(file_path_scaler)
    # load model
    file_path_model = os.path.join(script_dir, 'ANN_joblib_files',
                                   '501_Etc_Map_2226_Random_AlllAsTrain_save_01.joblib')
    final_model_Etc = load(file_path_model)

    # load map data
    data_path_map = 'Map_tiff_npy_files'
    filename_Tavg = 'world_input_Tavg.tif'
    filename_Tmin = 'world_input_Tmin.tif'
    filename_Tmax = 'world_input_Tmax.tif'

    file_full_Tavg = os.path.join(script_dir, data_path_map, filename_Tavg)
    file_full_Tmin = os.path.join(script_dir, data_path_map, filename_Tmin)
    file_full_Tmax = os.path.join(script_dir, data_path_map, filename_Tmax)

    # read tif files
    # Tavg
    src_Tavg = rasterio.open(file_full_Tavg)
    array_Tavg = src_Tavg.read(1)
    # Tmin
    src_Tmin = rasterio.open(file_full_Tmin)
    array_Tmin = src_Tmin.read(1)
    # Tmax
    src_Tmax = rasterio.open(file_full_Tmax)
    array_Tmax = src_Tmax.read(1)
    # lat
    height = array_Tavg.shape[0]
    width = array_Tavg.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src_Tavg.transform, rows, cols)
    array_lons = np.array(xs)
    array_lats = np.array(ys)

    file_full_GVI = os.path.join(script_dir, data_path_map, '204_GVI_MLP_2226_grid_save.py.npy')
    array_gvi = np.load(file_full_GVI)

    list_Tavg = array_Tavg.flatten()
    list_Tmin = array_Tmin.flatten()
    list_Tmax = array_Tmax.flatten()
    list_lats = array_lats.flatten()
    list_gvi = array_gvi.flatten()
    # add Low_Tsol and High_Tsol, B = np.full_like(A, 1)
    list_Low_Tsol = np.full_like(list_Tavg, target_values[1])
    list_High_Tsol = np.full_like(list_Tavg, target_values[0])
    list_Ttran = np.full_like(list_Tavg, target_values[2])

    nan_mask = np.isnan(list_gvi) | np.isnan(list_Tavg) | np.isnan(list_Tmin) | np.isnan(list_Tmax)
    list_lats[nan_mask] = np.nan

    data_map_input = pd.DataFrame({
        'abs_latitude': np.abs(list_lats),
        'gvi': list_gvi,
        'Tair_ave': list_Tavg,
        'Tair_min': list_Tmin,
        'Tair_max': list_Tmax,
        'Low_Tsol': list_Low_Tsol,
        'High_Tsol': list_High_Tsol,
        'TransitionTemperature': list_Ttran,
    })

    list_Etc = np.full_like(list_lats, np.nan)

    # map heatmap Etc
    data_map_input_scale_Etc = scaler.transform(data_map_input)
    list_Etc[~nan_mask] = final_model_Etc.predict(data_map_input_scale_Etc[~nan_mask])
    reshaped_array = list_Etc.reshape((height, width))

    # Save as a NumPy npy file
    np.save(f'{FileName}_Etc_TsolTtranAsInput.npy', reshaped_array)
    # Draw the wide heatmap occupying the entire subplot
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(reshaped_array, cmap='viridis', extent=[-180, 180, -55, 60], aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.grid(True, linestyle='dashed', linewidth=0.5, color='lightgrey')
    ax.set_xlabel('Longitude (\u00B0)')
    ax.set_ylabel('Latitude (\u00B0)')
    ax.set_title(
        f'$\u0394E_{{tc}}$ (kWh/m$^2$) by ANN, when $\u03C4_{{dark}}$ = {target_values[1]:.2f}, $\u03C4_{{clear}}$ = {target_values[0]:.2f}, and $T_{{tran}}$ = {target_values[2]:.2f}\u00B0C')
    plt.tight_layout()
    # Save the figure as a PNG file
    plt.savefig(f'{FileName}_Etc_TsolTtranAsInput.png', dpi=300)
    # Show the figure
    # plt.show()

    # Save as a GeoTIFF file
    with rasterio.open(
            f'{FileName}_Etc_TsolTtranAsInput.tif',
            'w',
            driver='GTiff',
            height=reshaped_array.shape[0],
            width=reshaped_array.shape[1],
            count=1,
            dtype=reshaped_array.dtype,
            crs=src_Tavg.crs,
            transform=src_Tavg.transform,
    ) as dst_Etc:
        dst_Etc.write(reshaped_array, 1)

# Function 06
# Output: En,
# Input: Tsol_dark, Tsol_clear, Ttran, FileName

def world_map_En_TsolTtranAsInput(Tsol_dark, Tsol_clear, Ttran, FileName):
    # input solar transmittance in clear state and dark state, and transition temperature
    target_values = np.array([Tsol_clear, Tsol_dark, Ttran])
    # load scaler
    # Get the current script directory
    script_dir = os.path.dirname(__file__)
    # Construct absolute path to the file you want to load
    file_path_scaler = os.path.join(script_dir, 'ANN_joblib_files',
                                    '502_En_Map_2226_Random_AlllAsTrain_save_scaler.joblib')
    scaler = load(file_path_scaler)
    # load model
    file_path_model = os.path.join(script_dir, 'ANN_joblib_files',
                                   '502_En_Map_2226_Random_AlllAsTrain_save_01.joblib')
    final_model_En = load(file_path_model)

    # load map data
    data_path_map = 'Map_tiff_npy_files'
    filename_Tavg = 'world_input_Tavg.tif'
    filename_Tmin = 'world_input_Tmin.tif'
    filename_Tmax = 'world_input_Tmax.tif'

    file_full_Tavg = os.path.join(script_dir, data_path_map, filename_Tavg)
    file_full_Tmin = os.path.join(script_dir, data_path_map, filename_Tmin)
    file_full_Tmax = os.path.join(script_dir, data_path_map, filename_Tmax)

    # read tif files
    # Tavg
    src_Tavg = rasterio.open(file_full_Tavg)
    array_Tavg = src_Tavg.read(1)
    # Tmin
    src_Tmin = rasterio.open(file_full_Tmin)
    array_Tmin = src_Tmin.read(1)
    # Tmax
    src_Tmax = rasterio.open(file_full_Tmax)
    array_Tmax = src_Tmax.read(1)
    # lat
    height = array_Tavg.shape[0]
    width = array_Tavg.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src_Tavg.transform, rows, cols)
    array_lons = np.array(xs)
    array_lats = np.array(ys)

    file_full_GVI = os.path.join(script_dir, data_path_map, '204_GVI_MLP_2226_grid_save.py.npy')
    array_gvi = np.load(file_full_GVI)

    list_Tavg = array_Tavg.flatten()
    list_Tmin = array_Tmin.flatten()
    list_Tmax = array_Tmax.flatten()
    list_lats = array_lats.flatten()
    list_gvi = array_gvi.flatten()
    # add Low_Tsol and High_Tsol, B = np.full_like(A, 1)
    list_Low_Tsol = np.full_like(list_Tavg, target_values[1])
    list_High_Tsol = np.full_like(list_Tavg, target_values[0])
    list_Ttran = np.full_like(list_Tavg, target_values[2])

    nan_mask = np.isnan(list_gvi) | np.isnan(list_Tavg) | np.isnan(list_Tmin) | np.isnan(list_Tmax)
    list_lats[nan_mask] = np.nan

    data_map_input = pd.DataFrame({
        'abs_latitude': np.abs(list_lats),
        'gvi': list_gvi,
        'Tair_ave': list_Tavg,
        'Tair_min': list_Tmin,
        'Tair_max': list_Tmax,
        'Low_Tsol': list_Low_Tsol,
        'High_Tsol': list_High_Tsol,
        'TransitionTemperature': list_Ttran,
    })

    list_En = np.full_like(list_lats, np.nan)

    # map heatmap En
    data_map_input_scale_En = scaler.transform(data_map_input)
    list_En[~nan_mask] = final_model_En.predict(data_map_input_scale_En[~nan_mask])
    reshaped_array = list_En.reshape((height, width))

    # Save as a NumPy npy file
    np.save(f'{FileName}_En_TsolTtranAsInput.npy', reshaped_array)
    # Draw the wide heatmap occupying the entire subplot
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(reshaped_array, cmap='viridis', extent=[-180, 180, -55, 60], aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.grid(True, linestyle='dashed', linewidth=0.5, color='lightgrey')
    ax.set_xlabel('Longitude (\u00B0)')
    ax.set_ylabel('Latitude (\u00B0)')
    ax.set_title(
        f'$\u0394E_{{n}}$ (kWh/m$^2$) by ANN, when $\u03C4_{{dark}}$ = {target_values[1]:.2f}, $\u03C4_{{clear}}$ = {target_values[0]:.2f}, and $T_{{tran}}$ = {target_values[2]:.2f}\u00B0C')
    plt.tight_layout()
    # Save the figure as a PNG file
    plt.savefig(f'{FileName}_En_TsolTtranAsInput.png', dpi=300)
    # Show the figure
    # plt.show()

    # Save as a GeoTIFF file
    with rasterio.open(
            f'{FileName}_En_TsolTtranAsInput.tif',
            'w',
            driver='GTiff',
            height=reshaped_array.shape[0],
            width=reshaped_array.shape[1],
            count=1,
            dtype=reshaped_array.dtype,
            crs=src_Tavg.crs,
            transform=src_Tavg.transform,
    ) as dst_En:
        dst_En.write(reshaped_array, 1)

# Function 07
# Output: TCRI,
# Input: Tsol_dark, Tsol_clear, Etc_npy_array, En_npy_array, FileName

def world_map_TCRI_TsolTtranAsInput(Tsol_dark, Tsol_clear, Ttran, Etc_npy_array, En_npy_array, FileName):
    # input solar transmittance in clear state and dark state
    target_values = np.array([Tsol_clear, Tsol_dark, Ttran])
    # Get the current script directory
    script_dir = os.path.dirname(__file__)
    # load Etc and En
    array_Etc = np.load(Etc_npy_array)
    array_En = np.load(En_npy_array)
    # array to list
    list_Etc = array_Etc.flatten()
    list_En = array_En.flatten()
    # ReLU
    list_Etc[list_Etc < 0] = 0
    list_En[list_En < 0] = 0

    # load Tavg
    data_path_map = 'Map_tiff_npy_files'
    filename_Tavg = 'world_input_Tavg.tif'
    file_full_Tavg = os.path.join(script_dir, data_path_map, filename_Tavg)
    # read tif files
    # Tavg
    src_Tavg = rasterio.open(file_full_Tavg)

    # define empty list for output Etc
    list_TCRI = list()
    # loop
    for value_Etc, value_En in zip(list_Etc, list_En):
        if (np.isnan(value_Etc) or np.isnan(value_En)):
            list_TCRI.append(np.nan)
        else:
            temp_input = value_Etc*value_En
            list_TCRI.append(temp_input)

    # reshape list
    reshaped_array = np.reshape(list_TCRI, (690, 2160))

    # Save as a NumPy npy file
    np.save(f'{FileName}_TCRI_TsolAsInput.npy', reshaped_array)
    # Draw the wide heatmap occupying the entire subplot
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(reshaped_array, cmap='viridis', extent=[-180, 180, -55, 60], aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.grid(True, linestyle='dashed', linewidth=0.5, color='lightgrey')
    ax.set_xlabel('Longitude (\u00B0)')
    ax.set_ylabel('Latitude (\u00B0)')
    ax.set_title(
        f'TCRI (kWh$^2$/m$^4$) by ANN, when $\u03C4_{{dark}}$ = {target_values[1]:.2f}, $\u03C4_{{clear}}$ = {target_values[0]:.2f}, and $T_{{tran}}$ = {target_values[2]:.2f}\u00B0C')
    plt.tight_layout()
    # Save the figure as a PNG file
    plt.savefig(f'{FileName}_TCRI_TsolAsInput.png', dpi=300)
    # Show the figure
    # plt.show()

    # Save as a GeoTIFF file
    with rasterio.open(
            f'{FileName}_TCRI_TsolAsInput.tif',
            'w',
            driver='GTiff',
            height=reshaped_array.shape[0],
            width=reshaped_array.shape[1],
            count=1,
            dtype=reshaped_array.dtype,
            crs=src_Tavg.crs,
            transform=src_Tavg.transform,
    ) as dst_TCRI:
        dst_TCRI.write(reshaped_array, 1)