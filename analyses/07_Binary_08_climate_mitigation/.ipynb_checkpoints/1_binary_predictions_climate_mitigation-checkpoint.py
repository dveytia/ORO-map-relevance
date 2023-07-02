import sys
print(sys.version)
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

# FOR TEST RUN
#rank_i = rank
rank_i = 0

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
binVar = "climate_mitigation" # name of binary variable to fit
codedVariablesTxt = '/home/dveytia/ORO-map-relevance/data/seen/all-coding-format-distilBERT-simplifiedMore.txt'
screenDecisionsTxt = '/home/dveytia/ORO-map-relevance/data/seen/all-screen-results_screenExcl-codeIncl.txt'
unseenTxt = '/home/dveytia/ORO-map-relevance/data/unseen/0_unique_references.txt' # change to unique_references2.txt?
relevanceTxt = '/home/dveytia/ORO-map-relevance/outputs/predictions-compiled/1_document_relevance_13062023.csv'


################ Load and format data #######################

# Load seen documents
seen_df = pd.read_csv(codedVariablesTxt, delimiter='\t')
seen_df = seen_df.rename(columns={'analysis_id':'id'})
seen_df['seen']=1

# Load unseen documents 
unseen_df = pd.read_csv(unseenTxt, delimiter='\t')
unseen_df = unseen_df.rename(columns={'analysis_id':'id'})
unseen_df=unseen_df.dropna(subset=['abstract']).reset_index(drop=True) # drop NA abstracts

# Load predictions of relevance (screen decisions)
pred_df = pd.read_csv(relevanceTxt)

# merge relevance predictions into unseen df
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


############# Define functions ########################

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

MODEL_NAME = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
   
def create_train_val(x,y,train,val):
    train_encodings = tokenizer(list(x[train].values),
                                truncation=True,
                                padding=True)
    val_encodings = tokenizer(list(x[val].values),
                                truncation=True,
                                padding=True) 
    
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        list(y[train].values)
    ))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        list(y[val].values)
    ))
    
    
    MAX_LEN = train_dataset._structure[0]['input_ids'].shape[0]
    
    return train_dataset, val_dataset, MAX_LEN

def init_model(MODEL_NAME, num_labels, params):
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)  
    optimizer = tfa.optimizers.AdamW(learning_rate=params['learning_rate'], weight_decay=params['weight_decay'])

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model
    
outer_scores = []
clfs = []

    
####################### Target label to predict ##################################
def train_eval_bert(params, df, train, test, evaluate = True):
    train_dataset, val_dataset, MAX_LEN = create_train_val(df['text'].astype("str"), df[binVar], train, test) #change here
    
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
    if evaluate:
        eps = evaluate_preds(df[binVar][test], y_pred[:,0]) #change here
        for key, value in params.items():
            eps[key] = value
        return eps, y_pred
    else:
        return y_pred

parallel=False

outer_scores = []
inner_scores = []
params = ['batch_size','weight_decay','learning_rate','num_epochs','class_weight']


###### Reads in results from model selection and chooses the best model ######
############## Change path to point to model_selection output ################
for k in range(3):
    inner_df = pd.read_csv(f'/home/dveytia/ORO-map-relevance/outputs/model_selection/{binVar}_model_selection_{k}.csv')
    inner_df = inner_df.sort_values('F1',ascending=False).reset_index(drop=True)
    inner_scores += inner_df.to_dict('records')

inner_scores = pd.DataFrame.from_dict(inner_scores).fillna(-1)
#inner_scores['F1 - tp'] = inner_scores.loc[:, [col for col in inner_scores.columns if col.startswith('F1 -') and any(target in col for target in targets)]].mean(axis=1)
best_model = (inner_scores
              .groupby(params)['F1']
              .mean()
              .sort_values(ascending=False)
              .reset_index() 
             ).to_dict('records')[0]


del best_model['F1']
print(best_model)
if best_model['class_weight']==-1:
    best_model['class_weight']=None
else:
    best_model['class_weight'] = ast.literal_eval(best_model['class_weight'])

########################## Runs model ###############################
##################### Change file paths x2 ##########################
outer_cv = KFold(n_splits=5)
for k, (train, test) in enumerate(outer_cv.split(seen_index)):    
    if k!=rank_i:
        continue
    train = seen_index[train]
    test = unseen_index

    y_preds = train_eval_bert(best_model, df=df, train=train, test=test, evaluate=False)
    
    np.save(f"/home/dveytia/ORO-map-relevance/outputs/predictions/{binVar}_y_preds_5fold_{k}.npz",y_preds) # Saves predictions

np.save(f"/home/dveytia/ORO-map-relevance/outputs/predictions/{binVar}_unseen_ids.npz",df.loc[unseen_index,"id"]) # Saves unseen ids 

print(t0 - time.time())


