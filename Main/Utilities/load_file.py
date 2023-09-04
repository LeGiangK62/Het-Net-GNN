import os
import scipy.io
import sys

if 'win' in sys.platform:
    DEFAULT_DATA_FOLDER = "..\..\Data"
else:
    DEFAULT_DATA_FOLDER = "../../Data"



def load_data_from_mat(file_path, data_folder=None):
    current_dir = os.path.dirname(__file__)
    if data_folder is None:
        data_folder = DEFAULT_DATA_FOLDER
    file_path = os.path.join(current_dir, data_folder, file_path)
    print(file_path )
    matLoader = scipy.io.loadmat(file_path)
    channelAll = matLoader['channel_python'].transpose(0, 2, 1)
    apSelectionAll = matLoader['mu_python'].transpose(0, 2, 1)
    powerAll = matLoader['power_python']
    EE_All = matLoader['EE_python']
    B = matLoader['B'][0][0]
    n0 = matLoader['n0'][0][0]
    num_ap = channelAll.shape[1]
    num_ue = channelAll.shape[2]
    num_sam = channelAll.shape[0]
    return channelAll, apSelectionAll, powerAll, EE_All, B, n0, (num_sam, num_ap, num_ue)

