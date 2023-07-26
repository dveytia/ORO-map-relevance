import sys
print(sys.version)
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

rank_j = rank%3

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
import time

t0 = time.time()


################# Change INPUTS ##################
targetVar = "m_co2_ocean_storage" # name of variable
codedVariablesTxt = '/home/dveytia/ORO-map-relevance/data/seen/all-coding-format-distilBERT-simplifiedMore.txt'
screenDecisionsTxt = '/home/dveytia/ORO-map-relevance/data/seen/all-screen-results_screenExcl-codeIncl.txt'
n_threads = 4 # number of threads to parallelize on


############################# Load data ###############################
######################## Change file paths x2 #########################
df = pd.read_csv(codedVariablesTxt, delimiter='\t') #File path 1

df = df.rename(columns={'analysis_id':'id'})

screendf = pd.read_csv(screenDecisionsTxt, delimiter='\t') #File path 2
screendf = screendf.query('include_screen==1')
screendf = screendf.rename(columns={'include_screen':'relevant','analysis_id':'id'})

df = df.merge(screendf[['id', 'sample_screen']], on='id', how='left')

def map_values(x):
    if x == "random":
        return 1
    elif x == "relevance sort":
        return 0
    elif x == "test list":
        return 0
    elif x == "supplemental coding":
        return 0
    else:
        return "NaN"

df['random_sample']=df['sample_screen'].apply(map_values)

df = (df
      #.query('unlabelled==0')
      # .query('relevant==1')
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)

df['text'] = df['title'] + ". " + df['abstract'] + " " + "Keywords: " + df["keywords"]
df['text'] = df.apply(lambda row: (row['title'] + ". " + row['abstract']) if pd.isna(row['text']) else row['text'], axis=1)

print("The data has been re-formatted")
print(df.shape)

######################### Define functions #############################

tf.config.threading.set_intra_op_parallelism_threads(n_threads)
tf.config.threading.set_inter_op_parallelism_threads(n_threads)

with open('/home/dveytia/ORO-map-relevance/pyFunctions/multi-label_0_model-selection_functions.py') as f:
    exec(f.read())

MODEL_NAME = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


###################### Select targets here #################################
targets = [x for x in df.columns if targetVar in x] 
df['labels'] = list(df[targets].values)

class_weight = {}
for i, t in enumerate(targets):
    cw = df[(df['random_sample']==1) & (df[t]==0)].shape[0] / df[(df['random_sample']==1) & (df[t]==1)].shape[0]
    class_weight[i] = cw
    
class_weight

bert_params = {
  "class_weight": [None,class_weight],
  "batch_size": [16, 32],
  "weight_decay": (0, 0.3),
  "learning_rate": (1e-5, 5e-5),
  "num_epochs": [2, 3, 4]
}


param_space = list(product_dict(**bert_params))

outer_cv = KFoldRandom(3, df.index, df[df['random_sample']!=1].index, discard=False)

outer_scores = []
clfs = []




############################## Run models ################################
######################## Change file path (x3) ###########################
for k, (train, test) in enumerate(outer_cv):    
    if k!=rank_j:
        continue
    try:
        pr = param_space[0]
        cv_results=pd.read_csv(f'/home/dveytia/ORO-map-relevance/outputs/model_selection/{targetVar}_model_selection_{k}.csv').to_dict('records') 
        params_tested=pd.read_csv(f'/home/dveytia/ORO-map-relevance/outputs/model_selection/{targetVar}_model_selection_{k}.csv')[list(pr.keys())].to_dict('records')
    except:
        cv_results = []
        params_tested = []
    for pr in param_space:
        if pr in params_tested:
            continue
        cv_results.append(train_eval_bert(pr, df=df, train=train, test=test))
        pd.DataFrame.from_dict(cv_results).to_csv(f'/home/dveytia/ORO-map-relevance/outputs/model_selection/{targetVar}_model_selection_{k}.csv',index=False) #File path + name 3 
        gc.collect()
