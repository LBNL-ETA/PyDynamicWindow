# This is an example of using three key parameters of thermo-responsive (TR) windows
# Solar transmittance in dark and clear states (Tsol_dark, Tsol_clear) and
# Transition temperature (Ttran) as inputs
# to generate global heatmaps of energy saving (Etr), necessity level (En), and TR recommendation index (TRRI)
# notice: not true map

from ANNs_TRwin_functions import world_map_Etr_TsolTtranAsInput, world_map_En_TsolTtranAsInput, world_map_TRRI_TsolTtranAsInput
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
    Etr_npy_array = f'{FileName}_Etr_TsolTtranAsInput.npy'
    En_npy_array = f'{FileName}_En_TsolTtranAsInput.npy'

    # generate world maps with functions
    world_map_Etr_TsolTtranAsInput(Tsol_dark, Tsol_clear, Ttran, FileName)
    world_map_En_TsolTtranAsInput(Tsol_dark, Tsol_clear, Ttran, FileName)
    # TRRI is calculated based on the results of Etr and En
    world_map_TRRI_TsolTtranAsInput(Tsol_dark, Tsol_clear, Ttran, Etr_npy_array, En_npy_array, FileName)

