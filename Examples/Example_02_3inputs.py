# This is an example of using three key parameters of thermochromic (TC) windows
# Solar transmittance in dark and clear states (Tsol_dark, Tsol_clear) and
# Transition temperature (Ttran) as inputs
# to generate global heatmaps of energy saving (Etc), necessity level (En), and TC recommendation index (TCRI)
# notice: not true map

from ANNs_TCwin_functions import world_map_Etc_TsolTtranAsInput, world_map_En_TsolTtranAsInput, world_map_TCRI_TsolTtranAsInput
#
if __name__ == '__main__':
    # ==========inputs begin==========
    # define the solar transmittance of thermochromic windows in dark and clear states, Tsol_dark, Tsol_clear
    # The range of Tsol_dark for training is [0, 0.3]
    # The range of Tsol_clear for training is [0.4, 0.8]
    # To obtain reliable results, it is better to ensure the inputs stay within the ranges
    Tsol_dark = 0.2
    Tsol_clear = 0.7
    Ttran = 25
    # define FileName
    FileName = 'Example_02_3inputs'
    # ==========inputs end==========

    # to be saved files that will be reused
    Etc_npy_array = f'{FileName}_Etc_TsolTtranAsInput.npy'
    En_npy_array = f'{FileName}_En_TsolTtranAsInput.npy'

    # generate world maps with functions
    world_map_Etc_TsolTtranAsInput(Tsol_dark, Tsol_clear, Ttran, FileName)
    world_map_En_TsolTtranAsInput(Tsol_dark, Tsol_clear, Ttran, FileName)
    # TCRI is calculated based on the results of Etc and En
    world_map_TCRI_TsolTtranAsInput(Tsol_dark, Tsol_clear, Ttran, Etc_npy_array, En_npy_array, FileName)

