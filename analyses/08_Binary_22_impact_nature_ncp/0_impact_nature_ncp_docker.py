import sys
print(sys.version)
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()


rank_j = rank%3
# FOR TEST RUN
#rank_j = 0

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

binVar = "impact_nature_ncp" # name of binary variable

dockerFilePath = '/home/devi/analysis/'
codedVariablesTxt = dockerFilePath + 'data/seen/all-coding-format-distilBERT-simplifiedMore.txt'
screenDecisionsTxt = dockerFilePath + 'data/seen/all-screen-results_screenExcl-codeIncl.txt'


################# Load data, change file path ################
df = pd.read_csv(codedVariablesTxt, delimiter='\t')
df = df.rename(columns={'analysis_id':'id'})

screendf = pd.read_csv(screenDecisionsTxt, delimiter='\t')
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
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)

df['text'] = df['title'] + ". " + df['abstract'] + " " + "Keywords: " + df["keywords"]
df['text'] = df.apply(lambda row: (row['title'] + ". " + row['abstract']) if pd.isna(row['text']) else row['text'], axis=1)


################# Merge impact_ncp.Any and impact_nature into one Binary response ###############

def merge_impact (row):
   if row['impact_ncp.Any'] == 1 :
      return 1
   if row['impact_nature'] == 1 :
      return 1
   return 0

df[binVar] = df.apply(lambda row: merge_impact(row), axis=1)


print("The data has been re-formatted")
print(df.shape)


#################### Define functions and parameters ##################

with open(dockerFilePath +'pyFunctions/binary-label_0_model-selection_functions.py') as f:
    exec(f.read())

tf.config.threading.set_intra_op_parallelism_threads(n_threads)
tf.config.threading.set_inter_op_parallelism_threads(n_threads)

MODEL_NAME = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


############# class weight, change target, e.g. "safe_space #################
cw = df[(df['random_sample']==1) & (df[binVar]==0)].shape[0] / df[(df['random_sample']==1) & (df[binVar]==1)].shape[0]
class_weight={0:1, 1:cw}

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

############## change target label ###############
def train_eval_bert(params, df, train, test):
    train_dataset, val_dataset, MAX_LEN = create_train_val(df['text'].astype("str"), df[binVar], train, test)
    
    print("training bert with these params")
    print(params)
    model = init_model('distilbert-base-uncased', 1, params)
    model.fit(train_dataset.shuffle(100).batch(params['batch_size']),
              epochs=params['num_epochs'],
              batch_size=params['batch_size'],
              class_weight=params['class_weight']
    )

    preds = model.predict(val_dataset.batch(1)).logits
    y_pred = tf.keras.activations.sigmoid(tf.convert_to_tensor(preds)).numpy()
    eps = evaluate_preds(df[binVar][test], y_pred[:,0])
    print(eps)
    for key, value in params.items():
        eps[key] = value
    return eps

#parallel=False

############################### Run model ####################################
############ Change file paths where output is stored x3 #####################


for k, (train, test) in enumerate(outer_cv):    
    if k!=rank_j:
        continue
    try:
        pr = param_space[0]
        cv_results=pd.read_csv(f'{dockerFilePath}outputs-docker/model_selection/{binVar}_model_selection_{k}.csv').to_dict('records') #File path 1, change name
        params_tested=pd.read_csv(f'{dockerFilePath}outputs-docker/model_selection/{binVar}_model_selection_{k}.csv')[list(pr.keys())].to_dict('records') #File path 2, change name
    except:
        cv_results = []
        params_tested = []
    for pr in param_space:
        if pr in params_tested:
            continue
        cv_results.append(train_eval_bert(pr, df=df, train=train, test=test))
        pd.DataFrame.from_dict(cv_results).to_csv(f'{dockerFilePath}outputs-docker/model_selection/{binVar}_model_selection_{k}.csv',index=False) #File path 3, change name
        gc.collect()
        

                         