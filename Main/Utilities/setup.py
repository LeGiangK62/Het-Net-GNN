import argparse
from pathlib import Path


def get_arguments():
    parser = argparse.ArgumentParser()

    """ ===================================================================== """
    """ =========================== SYSTEM CONFIG =========================== """
    """ ===================================================================== """

    parser.add_argument("--user_num", type=int, default=10,
                        help="number of users (UEs)")
    parser.add_argument("--ap_num", type=int, default=3,
                        help="number of access points (APs)")
    parser.add_argument("--noise", type=float, default=10e-12,
                        help="network noise")
    parser.add_argument("--radius", type=float, default=300,
                        help="network area radius")
    parser.add_argument("--poweru_max", type=float, default=200,
                        help="max power of user threshold (mW)")
    parser.add_argument("--power_cir", type=float, default=200,
                        help="circuit power of user (mW)")
    parser.add_argument("--bandwidth", type=float, default=180000,
                        help="signal bandwidth (Hz)")
    """==== TEST PART ===="""
    parser.add_argument("--user_num_test", type=int, default=0,
                        help="number of users (UEs) - for Testing")
    parser.add_argument("--ap_num_test", type=int, default=0,
                        help="number of access points (APs) - for Testing")
    parser.add_argument("--noise_test", type=float, default=10e-12,
                        help="network noise - for Testing")
    parser.add_argument("--radius_test", type=float, default=0,
                        help="network area radius - for Testing")
    parser.add_argument("--poweru_max_test", type=float, default=200,
                        help="max power of user threshold (mW) - for Testing")
    parser.add_argument("--power_cir_test", type=float, default=200,
                        help="circuit power of user (mW) - for Testing")
    parser.add_argument("--bandwidth_test", type=float, default=180000,
                        help="signal bandwidth (Hz) - for Testing")

    """ ===================================================================== """
    """ ========================== Hyper Parameters ========================= """
    """ ===================================================================== """

    parser.add_argument("--batch_size", type=int, default=64,
                        help="data batch size")
    parser.add_argument("--epoch_num", type=int, default=200,
                        help="number of epochs")
    parser.add_argument("--per_epoch", type=int, default=50,
                        help="number of epochs per plot result")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument("--train_num", type=int, default=200,
                        help="number of training samples")
    parser.add_argument("--test_num", type=int, default=100,
                        help="number of testing samples")

    """ ===================================================================== """
    """ ========================== Model Parameters ========================= """
    """ ===================================================================== """
    parser.add_argument("--model_mode", type=str, default="withAP",
                        help="withAP | withoutAP")
    parser.add_argument("--loss_type", type=str, default="GlobalEE",
                        help="GlobalEE | SumEE | ProdEE")
    parser.add_argument("--generate_data", type=bool, default=True,
                        help="Auto generate Data (True) | Use Input (False)")
    parser.add_argument("--train_file", type=str, default="blank",
                        help="Name of Data File for training (inside Main folder)")
    parser.add_argument("--test_file", type=str, default="blank",
                        help="Name of Data File for testing (inside Main folder)")
    return parser.parse_args()