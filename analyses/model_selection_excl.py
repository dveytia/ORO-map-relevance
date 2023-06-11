import sys
print(sys.version)
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

rank_i = rank%5

import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import precision_score, recall_score
import itertools
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import tensorflow_addons as tfa

df = pd.read_csv('0_all-screen-results_screenExcl-codeIncl.txt', delimiter='\t')
#df = pd.read_csv('C:\\Users\\vcm20gly\\OneDrive - Bangor University\\Documents\\Review\\0_all-screen-results_screenExcl-codeIncl.txt', delimiter='\t')

df = df.rename(columns={'include_screen':'relevant','analysis_id':'id'})
df['relevant']=df['relevant'].astype(int)

def map_values(x):
    value_map = {
        "random": 1,
        "relevance sort": 0,
        "test list": 0,
        "supplemental coding": 0
    }
    return value_map.get(x, "NaN")

df['random_sample']=df['sample_screen'].apply(map_values)

df = (df
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)

df['text'] = df['title'] + ". " + df['abstract'] + " " + "Keywords: " + df["keywords"]
df['text'] = df.apply(lambda row: (row['title'] + ". " + row['abstract']) if pd.isna(row['text']) else row['text'], axis=1)

print("The data has been re-formatted")

def KFoldRandom(n_splits, X, no_test, shuffle=False, discard=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    for train, test in kf.split(X):
        if not discard: #Discard = True, not value would be false and staement won't run
            train = list(train) +  [x for x in test if x in no_test]
        test = [x for x in test if x not in no_test]
        yield (train, test)


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

def evaluate_preds(y_true, y_pred):
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except:
        roc_auc = np.NaN
    f1 = f1_score(y_true, y_pred.round())
    p, r = precision_score(y_true, y_pred.round()), recall_score(y_true, y_pred.round())
    acc = accuracy_score(y_true, y_pred.round())
    print(f"ROC AUC: {roc_auc:.0%}, F1: {f1:.1%}, precision: {p:.1%}, recall {r:.1%}, acc {acc:.0%}")
    return {"ROC AUC": roc_auc, "F1": f1, "precision": p, "recall": r, "accuracy": acc}


cw = df[(df['random_sample']==1) & (df['relevant']==0)].shape[0] / df[(df['random_sample']==1) & (df['relevant']==1)].shape[0]
class_weight={0:1, 1:cw}

bert_params = {
  "class_weight": [None,class_weight],
  "batch_size": [16, 32],
  "weight_decay": (0, 0.3),
  "learning_rate": (1e-5, 5e-5),
  "num_epochs": [2, 3, 4]
}


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
            
param_space = list(product_dict(**bert_params))
len(param_space)

outer_cv = KFoldRandom(5, df.index, df[df['random_sample']!=1].index, discard=False)

outer_scores = []
clfs = []


def train_eval_bert(params, df, train, test):
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
    eps = evaluate_preds(df['relevant'][test], y_pred[:,0])  
    for key, value in params.items():
        eps[key] = value
    return eps

parallel=False

for k, (train, test) in enumerate(outer_cv):    
    if k!=rank_i:
        continue
    try:
        pr = param_space[0]
        cv_results=pd.read_csv(f'model_selection_excl/model_selection_{rank_i}.csv').to_dict('records')
        params_tested=pd.read_csv(f'model_selection_excl/model_selection_{rank_i}.csv')[list(pr.keys())].to_dict('records')
    except:
        cv_results = []
        params_tested = []

    for pr in param_space:
        if pr in params_tested:
            continue
        cv_results.append(train_eval_bert(pr, df=df, train=train, test=test))
        pd.DataFrame.from_dict(cv_results).to_csv(f'model_selection_excl/model_selection_{rank_i}.csv',index=False)
        gc.collect()
