import sys
print(sys.version)
#import mpi4py
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

rank_j = rank%3

!python -m pip install sklearn

import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold