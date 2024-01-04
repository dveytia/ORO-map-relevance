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


################# Change INPUTS ##################
n_threads = 8 # number of threads to parallelize on
dockerFilePath = '/home/devi/analysis/'

targetVar = "method_type" # name of variable 
suffix = "simplified" # the suffix to add to this run of the variable 
conditionVar = ["data_type.Primary"] # the variable(s) that has to ==1 in order to predict the target Var
dropLabels = ['method_type.Expert_opinion', 'method_type.Method_or_technology_development']
mergeLabels= ['method_type.Experimental_exsitu', 'method_type.Experimental_insitu', 'method_type.Observational'] # list of labels to merge
newLabel = 'method_type.Empirical' # name of new label that encompasses the merged labels

codedVariablesTxt = dockerFilePath + 'data/seen/all-coding-format-distilBERT-simplifiedMore.txt'
screenDecisionsTxt = dockerFilePath + 'data/seen/all-screen-results_screenExcl-codeIncl.txt'



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


######### PREDICT | CONDITIONAL VARIABLE == 1 #################
############################ Choose subset (nested) ##########################
df = df[df[conditionVar].sum(axis=1) == len(conditionVar)] # keep rows where all conditional variables = 1
df = df.drop(columns=conditionVar) # drop the conditional variable name otherwise it will be fit along with the other impact_ncp labels

df = (df
      #.query('unlabelled==0')
      # .query('relevant==1')
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)




################# Merge variables to simplify labels of predicted variable ###############
# Merge into one column
df[newLabel] = df[mergeLabels].sum(axis = 'columns') # get the sum across all the columns to merge
df[newLabel].where(df[newLabel] <= 1, 1) # cap value at 1

# Remove old columns
df = df.drop(columns=mergeLabels)
df = df.drop(columns = dropLabels)

print("The data has been re-formatted")
print(df.shape)



######################### Define functions #############################

tf.config.threading.set_intra_op_parallelism_threads(n_threads)
tf.config.threading.set_inter_op_parallelism_threads(n_threads)

with open(dockerFilePath + 'pyFunctions/multi-label_0_model-selection_functions.py') as f:
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
        cv_results=pd.read_csv(f'{dockerFilePath}outputs-docker/model_selection/{targetVar}_{suffix}_model_selection_{k}.csv').to_dict('records') #File path 1, change name
        params_tested=pd.read_csv(f'{dockerFilePath}outputs-docker/model_selection/{targetVar}_{suffix}_model_selection_{k}.csv')[list(pr.keys())].to_dict('records') #File path 2, change name
    except:
        cv_results = []
        params_tested = []
    for pr in param_space:
        if pr in params_tested:
            continue
        cv_results.append(train_eval_bert(pr, df=df, train=train, test=test))
        pd.DataFrame.from_dict(cv_results).to_csv(f'{dockerFilePath}outputs-docker/model_selection/{targetVar}_{suffix}_model_selection_{k}.csv',index=False) #File path 3, change name
        gc.collect()
        
   

                         