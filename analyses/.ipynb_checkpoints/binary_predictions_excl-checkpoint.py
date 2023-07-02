# for test run, comment these out
#import sys
#print(sys.version)
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#num_procs = comm.Get_size()
#rank = comm.Get_rank()
#rank_i = rank

# instead run this line:
rank_i = 0

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import ast
import time

t0 = time.time()

# Change the name of the documents here:
seen_df = pd.read_csv('/home/dveytia/ORO-map-relevance/data/seen/all-screen-results_screenExcl-codeIncl.txt', delimiter='\t')
seen_df['seen']=1
seen_df = seen_df.rename(columns={'include_screen':'relevant','analysis_id':'id'})
seen_df['relevant']=seen_df['relevant'].astype(int)

def map_values(x): 
    value_map = {
        "random": 1,
        "relevance sort": 0,
        "test list": 0,
        "supplemental coding": 0
    }
    return value_map.get(x, "NaN")

seen_df['random_sample'] = seen_df['sample_screen'].apply(map_values)

# add this in to sub-sample dataframe for test run
seen_df1 = seen_df[(seen_df['sample_screen'] == 'random') & (seen_df['relevant'] == 1)]
seen_df1 = seen_df1[0:15]
seen_df2 = seen_df[(seen_df['sample_screen'] == 'random') & (seen_df['relevant'] == 0)]
seen_df2 = seen_df2[0:15]
seen_df = seen_df1.append(seen_df2).reset_index(drop=True)

unseen_df = pd.read_csv('/home/dveytia/ORO-map-relevance/data/unseen/unique_references2.txt', delimiter='\t')
# unseen_df = pd.read_csv('C:\\Users\\vcm20gly\\OneDrive - Bangor University\\Documents\\Review\\0_unique_unlabelled_references_VM.csv')
unseen_df.rename(columns={'analysis_id':'id'}, inplace=True)
unseen_df['seen']=0
unseen_df = unseen_df[0:30] # add this to subsample df for test run

nan_count=unseen_df['abstract'].isna().sum()
print('Number of missing abstracts is',nan_count)
nan_articles=unseen_df[unseen_df['abstract'].isna()]
unseen_df=unseen_df.dropna(subset=['abstract']).reset_index(drop=True)

df = (pd.concat([seen_df,unseen_df])
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)

print('Number of unique references WITH abstract is',len(df))

#df['text'] = df['title'] + ". " + df['abstract'] + " " + "Keywords: " + df["keywords"] 
# sometimes this line above throws an error, so if it does, run:
df['text'] = df['title'].astype("str") + ". " + df['abstract'].astype("str") + " " + "Keywords: " + df["keywords"].astype("str") 

df['text'] = df.apply(lambda row: (row['title'] + ". " + row['abstract']) if pd.isna(row['text']) else row['text'], axis=1)

seen_index = df[df['seen']==1].index
unseen_index = df[df['seen']==0].index

from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import tensorflow_addons as tfa

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


def train_eval_bert(params, df, train, test, evaluate = True):
    train_dataset, val_dataset, MAX_LEN = create_train_val(df['text'].astype("str"), df['relevant'], train, test)
    
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
        eps = evaluate_preds(df['relevant'][test], y_pred[:,0])  
        for key, value in params.items():
            eps[key] = value
        return eps, y_pred
    else:
        return y_pred

parallel=False

outer_scores = []
inner_scores = []
params = ['batch_size','weight_decay','learning_rate','num_epochs','class_weight']



## ERROR:
# FileNotFoundError: [Errno 2] No such file or directory: 'model_selection_excl/model_selection_0.csv'
# I think it needs the output from model_selection_excl.py in order to run

# Reads in results from model selection and chooses the best model
for k in range(1): # changed from range(5) for test run
    inner_df = pd.read_csv(f'/home/dveytia/ORO-map-relevance/outputs/model_selection/model_selection_{k}.csv') 
    inner_df = inner_df.sort_values('F1',ascending=False).reset_index(drop=True)
    inner_scores += inner_df.to_dict('records')

inner_scores = pd.DataFrame.from_dict(inner_scores).fillna(-1)
best_model = (inner_scores
              .groupby(params)['F1']
              .mean()
              .sort_values(ascending=False)
              .reset_index() 
             ).to_dict('records')[0]


# can have a look at the F1 score for the best model
print(best_model)

del best_model['F1']
print(best_model)

if best_model['class_weight']==-1:
    best_model['class_weight']=None
else:
    best_model['class_weight'] = ast.literal_eval(best_model['class_weight'])

outer_cv = KFold(n_splits=10)
for k, (train, test) in enumerate(outer_cv.split(seen_index)):    
    if k!=rank_i:
        continue
    train = seen_index[train]
    test = unseen_index

    y_preds = train_eval_bert(best_model, df=df, train=train, test=test, evaluate=False)
    
    np.save(f"/home/dveytia/ORO-map-relevance/outputs/predictions/y_preds_10fold_{k}.npz",y_preds) # Saves predictions

np.save("/home/dveytia/ORO-map-relevance/outputs/predictions/unseen_ids.npz",df.loc[unseen_index,'id']) # these are the unseen ids

print(t0 - time.time())
