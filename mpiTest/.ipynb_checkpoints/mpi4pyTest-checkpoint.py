from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import precision_score, recall_score
import itertools


rank_j = rank%3

pd.DataFrame.from_dict({'col_one':[2,3],'col_two':[4,5]}).to_csv(f'/home/devi/analysis/outputs-docker/mpiTest/mpiTest_{rank_j}.csv',index=False)


print('Hello from processor {} of {}'.format(rank_j,size))
