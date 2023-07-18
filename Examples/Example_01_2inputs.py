# This is an example of using two solar transmittances of thermochromic (TC) windows (Tsol_dark, Tsol_clear)
# to generate global heatmaps of energy saving (Etc), necessity level (En), TC recommendation index (TCRI), and
# optimal transition temperature (Ttran)
# notice: not true map

from ANNs_TCwin_functions import world_map_Etc_TsolAsInput, world_map_En_TsolAsInput, world_map_Ttran_TsolAsInput, world_map_TCRI_TsolAsInput

#
if __name__ == '__main__':
    # ==========inputs begin==========
    # define the solar transmittance of thermochromic windows in dark and clear states, Tsol_dark, Tsol_clear
    # The range of Tsol_dark for training is [0, 0.3]
    # The range of Tsol_clear for training is [0.4, 0.8]
    # To obtain reliable results, it is better to ensure the inputs stay within the ranges
    Tsol_dark = 0.2
    Tsol_clear = 0.7
    # define FileName
    FileName = 'Example_01_2inputs'
    # ==========inputs end==========

    # to be saved files that will be reused
    Etc_npy_array = f'{FileName}_Etc_TsolAsInput.npy'
    En_npy_array = f'{FileName}_En_TsolAsInput.npy'

    # generate world maps with functions
    world_map_Etc_TsolAsInput(Tsol_dark, Tsol_clear, FileName)
    world_map_En_TsolAsInput(Tsol_dark, Tsol_clear, FileName)
    world_map_Ttran_TsolAsInput(Tsol_dark, Tsol_clear, FileName)
    # TCRI is calculated based on the results of Etc and En
    world_map_TCRI_TsolAsInput(Tsol_dark, Tsol_clear, Etc_npy_array, En_npy_array, FileName)

