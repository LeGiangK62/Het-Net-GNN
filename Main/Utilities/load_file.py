import scipy.io


def load_data_from_mat(file_path):
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
