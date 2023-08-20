import argparse
import os
import time

from Utilities.setup import *
from Main.HetNet_AP import *

if __name__ == "__main__":
    total_start = time.time()

    args = get_arguments()

    training_loss, testing_acc = main_train(args)

    # print(training_loss)


