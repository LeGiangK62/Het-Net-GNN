import os
import scipy.io
import sys

if 'win' in sys.platform:
    DEFAULT_DATA_FOLDER = "..\..\Data"
else:
    DEFAULT_DATA_FOLDER = "../../Data"


# Get data from pre-save .mat file
def load_data_from_mat(file_path, data_folder=None):
    current_dir = os.path.dirname(__file__)
    if data_folder is None:
        file_path = os.path.join(current_dir, DEFAULT_DATA_FOLDER, file_path)
    else:
        file_path = os.path.join(current_dir, "..\..", data_folder, file_path)
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


def loading_data(num_train, num_test, channel_load, theta_load, power, EE_load, noise):
    X_train, theta_train, noise_train = channel_load[0:num_train]**2/noise, theta_load[0:num_train], 1
    EE_train = EE_load[0:num_train]
    X_test, theta_test, noise_test = channel_load[num_train:(num_train + num_test)]**2/noise, theta_load[num_train:(num_train + num_test)], 1
    EE_test = EE_load[num_train:(num_train + num_test)]

    return (X_train, theta_train, noise_train, EE_train), (X_test, theta_test, noise_test, EE_test)