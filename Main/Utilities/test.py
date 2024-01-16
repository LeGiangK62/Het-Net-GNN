import numpy as np
from load_file import load_data_from_mat_v2


# test_file = 'density_200_200_1000_500_compact_17Sept.mat'
test_file = 'pmax_10_40_200_500_compact_17Sept.mat'

channel_test, theta_test, power_test, EE_result_test, bandW_test, noise_test, \
                (num_set_up_test, num_s_test, num_aps_test, num_ues_test) = load_data_from_mat_v2(test_file)

# print(f'channel_test: {channel_test.shape}')
# print(f'theta_test: {theta_test.shape}')
# print(f'power_test: {power_test.shape}')
# print(f'EE_result_test: {EE_result_test.shape}')
# print(f'bandW_test: {bandW_test}')
# print(f'noise_test: {noise_test}')
# print(f'size: {num_set_up_test, num_s_test, num_aps_test, num_ues_test}')


print(EE_result_test)
