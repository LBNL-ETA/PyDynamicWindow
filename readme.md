# PyDynamicWindow

## Introduction
This module provides an open-source tool to optimize the intrinsic properties of thermo-responsive (TR) dynamic windows and evaluate their energy performance in building windows on a global scale. TR dynamic windows can be realized by incorporating a layer of thermochromic (TC) material into the window glazing or by integrating temperature sensors with electrochromic (EC) windows. Regaring building energy saving, three key parameters of TR windows include:
1. dark-state solar transmittance ($\tau_{dark}$), 
2. dark-state solar transmittance ($\tau_{clear}$), and 
3. transition temperature ($T_{tran}$).

Three indicators are used to evaluate TR windows:
1. Energy saving ($\Delta E_{TR}$)
2. Necessity level ($\Delta E_{n}$)
3. TR recommendation index (TRRI)

![Definitions of indicators for TR windows](images/Figure_definitions.png)

This tool can be used mainly in two scenarios:
1. Optimize $T_{tran}$ by given $\tau_{dark}$ and $\tau_{clear}$, and evaluate energy performance (Example 01)
2. Evaluate energy performance by given $\tau_{dark}$, $\tau_{clear}$, and $T_{tran}$ (Example 02)

The artificial neural network models are trained by EnergyPlus simulation results using weather files of over two thousand global locations. The EnergyPlus model is adapted from Department of Energy's (DOE's) prototype building model for a medium-size office building. Technical details can be found in [our publication](https://www.nature.com/articles/s41467-024-54967-8). Please cite this [publication](https://www.nature.com/articles/s41467-024-54967-8) if you use this tool in your article.

For other questions, please contact Dr. Yuan Gao - y.gao@lbl.gov

## System Requirements
### Software Dependencies
* Python 3.9 or higher
* See [requirements.txt](https://github.com/LBNL-ETA/PyDynamicWindow/blob/main/requirements.txt)

### Operating Systems
* Windows 10 or higher
* macOS Catalina 10.15 or higher

### Tested Versions
* Windows 10, Python 3.11
* macOS Sonoma 14.3.1, Python 3.9

### Required Non-Standard Hardware
* No non-standard hardware required.

## Installation and Demo
See a [tutorial video](https://drive.google.com/file/d/15SkSaynakWd4mJWn6N0924oiRJLTbo21/view?usp=drive_link)
* For Python beginners, PyCharm is used in this tutorial video.
* First, download this python package. Go to "Code", "Download ZIP". Unzip it to your computer.
* Second, go to [WorldClim website](https://www.worldclim.org/data/worldclim21.html) to download "tmin 10m", "tmax 10m", and "tavg 10m" zip files and unzip them to the "Map_tiff_npy_files" folder (see the tutorial video for details). Then, run [data_prep_resize_geotif_Tmp.py](https://github.com/LBNL-ETA/PyDynamicWindow/blob/main/data_prep_resize_geotif_Tmp.py) and generate three tif files in the "Map_tiff_npy_files" folder. This step is important. The PyDynamicWindow tool cannot be used without the generated world temperature tif files due to the lack of inputs. The reason we don't have the files ready here is because of license issues.
* Third, In PyCharm, go to "Tools", "Sync Python Requirements" to install all the required packages. After this step, the PyDynamicWindow tool is ready to be used.
* Go to the "Examples" folder. Run "Example_01_2inputs.py" or "Example_01_3inputs.py" by changing the properties of TR windows.

* Typical install time on a "normal" desktop computer: 2 - 5 minutes
* Expected output: world map data of $\Delta E_{TR}$, $\Delta E_{n}$, TRRI, and optimial $T_{tran}$ in the format of npy, png, and tif.
* Expected run time for demo on a "normal" desktop computer: 10 - 30 seconds

## License
See [license file](https://github.com/LBNL-ETA/PyDynamicWindow/blob/main/license.txt) for details

*** Copyright Notice ***

PyDynamicWindow Copyright (c) 2024, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
