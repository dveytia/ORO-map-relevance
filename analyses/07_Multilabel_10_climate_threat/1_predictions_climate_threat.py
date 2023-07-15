import sys
print(sys.version)
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

rank_i = rank%5

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import ast
import time
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import tensorflow_addons as tfa

t0 = time.time()


################# Change INPUTS ##################
targetVar = "climate_threat" # name of variable
codedVariablesTxt = '/home/dveytia/ORO-map-relevance/data/seen/all-coding-format-distilBERT-simplifiedMore.txt'
screenDecisionsTxt = '/home/dveytia/ORO-map-relevance/data/seen/all-screen-results_screenExcl-codeIncl.txt'
unseenTxt = '/home/dveytia/ORO-map-relevance/data/unseen/0_unique_references.txt' # change to unique_references2.txt?
relevanceTxt = '/home/dveytia/ORO-map-relevance/outputs/predictions-compiled/1_document_relevance_13062023.csv'
n_threads = 3 # number of threads to parallelize on


############################# Load data ###############################
######################## Change file paths x3 #########################
# Load seen documents
seen_df = pd.read_csv(codedVariablesTxt, delimiter='\t') 
seen_df = seen_df.rename(columns={'analysis_id':'id'})
seen_df['seen']=1

# Load unseen documents and merge
unseen_df = pd.read_csv(unseenTxt, delimiter='\t') 
unseen_df = unseen_df.rename(columns={'analysis_id':'id'})
unseen_df=unseen_df.dropna(subset=['abstract']).reset_index(drop=True)

# Load prediction relevance
pred_df = pd.read_csv(relevanceTxt) 

unseen_df = unseen_df.merge(pred_df, how="left")
unseen_df['seen']=0

# Choose which predictiction boundaries to apply
unseen_df = unseen_df[unseen_df['0 - relevance - upper_pred']>=0.5]


# Concatenate seen and unseen
df = (pd.concat([seen_df,unseen_df])
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)

df['text'] = df['title'].astype("str") + ". " + df['abstract'].astype("str") + " " + "Keywords: " + df["keywords"].astype("str")
df['text'] = df.apply(lambda row: (row['title'] + ". " + row['abstract']) if pd.isna(row['text']) else row['text'], axis=1)


seen_index = df[df['seen']==1].index
unseen_index = df[df['seen']==0].index

print("Dataset has been re-formatted and is ready")

################ Start defining functions ############################
tf.config.threading.set_intra_op_parallelism_threads(n_threads)
tf.config.threading.set_inter_op_parallelism_threads(n_threads)

MODEL_NAME = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

with open('/home/dveytia/ORO-map-relevance/pyFunctions/multi-label_1_predictions_functions.py') as f:
    exec(f.read())

##################### Select targets here ###########################
targets = [x for x in df.columns if targetVar in x] #Only need to change here, "data_type" for another variable
df['labels'] = list(df[targets].values)

class_weight = {}
try:
    for i, t in enumerate(targets):
        cw = df[(df['random_sample']==1) & (df[t]==0)].shape[0] / df[(df['random_sample']==1) & (df[t]==1)].shape[0]
        class_weight[i] = cw
except:
    class_weight=None

outer_scores = []
clfs = []


parallel=False

################### Load best model (change file paths!) #####################
outer_scores = []
inner_scores = []
params = ['batch_size','weight_decay','learning_rate','num_epochs','class_weight']

for k in range(3):
    inner_df = pd.read_csv(f'/home/dveytia/ORO-map-relevance/outputs/model_selection/{targetVar}_model_selection_{k}.csv') 
    inner_df = inner_df.sort_values('F1 macro',ascending=False).reset_index(drop=True)
    inner_scores += inner_df.to_dict('records')

inner_scores = pd.DataFrame.from_dict(inner_scores).fillna(-1)
inner_scores['F1 - tp'] = inner_scores.loc[:, [col for col in inner_scores.columns if col.startswith('F1 -') and any(target in col for target in targets)]].mean(axis=1)

best_model = (inner_scores
              .groupby(params)['F1 - tp']
              .mean()
              .sort_values(ascending=False)
              .reset_index() 
             ).to_dict('records')[0]

del best_model['F1 - tp']
print(best_model)
if best_model['class_weight']==-1:
    best_model['class_weight']=None
else:
    best_model['class_weight'] = ast.literal_eval(best_model['class_weight'])


######################### Run model #######################################
##################### Change paths x2 #####################################
outer_cv = KFold(n_splits=5)
for k, (train, test) in enumerate(outer_cv.split(seen_index)):    
    if k!=rank_i:
        continue
    train = seen_index[train]
    test = unseen_index

    y_preds = train_eval_bert(best_model, df=df, train=train, test=test, evaluate=False)
   
    np.save(f'/home/dveytia/ORO-map-relevance/outputs/predictions/{targetVar}_y_preds_5fold_data_{k}.npz',y_preds) #Change file path + name

np.save(f'/home/dveytia/ORO-map-relevance/outputs/predictions_data/{targetVar}_data_pred_ids',df.loc[unseen_index,"id"]) #Change file path + name

print(t0 - time.time())
