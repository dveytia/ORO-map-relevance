## branch_huggingfaceModelCard

# Load modules
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import pandas as pd
import tensorflow as tf
a = tf.zeros([], tf.float32)
import tensorflow_addons as tfa
from datasets import load_dataset, Dataset, DatasetDict

# Set up variables
dockerFilePath = '/home/devi/analysis'

# Define which model, tokenizer that will be used to fit the model
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


# Collect all the parameters from the model selection
inner_scores = []
params = ['batch_size','weight_decay','learning_rate','num_epochs','class_weight']

for k in range(3):
    inner_df = pd.read_csv(f'{dockerFilePath}/outputs/model_selection/oro_branch_model_selection_{k}.csv')
    inner_df = inner_df.sort_values('F1 macro',ascending=False).reset_index(drop=True)
    inner_scores += inner_df.to_dict('records')
    
    
# From all the folds used for model selection, find the best model
inner_scores = pd.DataFrame.from_dict(inner_scores).fillna(-1)
inner_scores['F1 - tp'] = inner_scores.loc[:, [col for col in inner_scores.columns if col.startswith('F1 -')]].mean(axis=1) #and any(target in col for target in targets)

best_model_params = (inner_scores
              .groupby(params)['F1 - tp'] # This is the same as groupig through F1-macro
              .mean()
              .sort_values(ascending=False)
              .reset_index() 
             ).to_dict('records')[0]

del best_model_params['F1 - tp']
print(best_model_params)

if best_model_params['class_weight']==-1:
    best_model_params['class_weight']=None
else:
    best_model_params['class_weight'] = ast.literal_eval(best_model_params['class_weight'])


    
## Using the best parameters, fit the model on the full seen dataset

## Read in and Format the oro_branch coding data 
## The 'seen' data
codedVariablesTxt = dockerFilePath + '/data/seen/all-coding-format-distilBERT-simplifiedMore.txt'
screenDecisionsTxt = dockerFilePath + '/data/seen/all-screen-results_screenExcl-codeIncl.txt'

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


## Format target columns as labels for the model
targets = [x for x in df.columns if "oro_branch" in x] #Only need to change here, "data_type" for another variable
print(targets)

df['labels'] = list(df[targets].values)
print(df['labels'].head())


## Convert pandas data frame to Dataset
## separate into training (non-randomly sampled) and testing (randomly sampled)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


train_datasets = Dataset.from_pandas(df.loc[df['random_sample'] == 0, ['text','labels','random_sample']])
eval_datasets = Dataset.from_pandas(df.loc[df['random_sample'] == 1, ['text','labels','random_sample']])

train_tokenized = train_datasets.map(tokenize_function, batched=True)
eval_tokenized = eval_datasets.map(tokenize_function, batched=True)


# Convert Dataset to big tensors and use the tf.data.Dataset.from_tensor_slices method
full_train_dataset = train_tokenized
full_eval_dataset = eval_tokenized

tf_train_dataset = full_train_dataset.remove_columns(["text"]).with_format("tensorflow")
train_features = {x: tf_train_dataset[x] for x in tokenizer.model_input_names}
train_tf_dataset = tf.data.Dataset.from_tensor_slices((train_features, tf_train_dataset['labels']))
train_tf_dataset = train_tf_dataset.shuffle(len(tf_train_dataset)).batch(2) ## reduce batch size

tf_eval_dataset = full_eval_dataset.remove_columns(["text"]).with_format("tensorflow")
eval_features = {x: tf_eval_dataset[x] for x in tokenizer.model_input_names}
eval_tf_dataset = tf.data.Dataset.from_tensor_slices((eval_features, tf_eval_dataset['labels']))
eval_tf_dataset = eval_tf_dataset.shuffle(len(tf_eval_dataset)).batch(2) ## reduce batch size


# With this, the model can be compiled and trained 
# define model using best parameters gotten from model selection
num_labels = 3 # three oro branch labels
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', 
                                                              num_labels=num_labels,
                                                              id2label={0: 'Mitigation', 1: 'Natural', 2:'Societal'})  
optimizer = tfa.optimizers.AdamW(learning_rate=best_model_params['learning_rate'], weight_decay=best_model_params['weight_decay'])
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)
# Fit model using training and evaluation datasets
model.fit(train_tf_dataset, validation_data=eval_tf_dataset, epochs=best_model_params['num_epochs']),


model.push_to_hub("distilbert_ORO_Branch", use_auth_token = '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')
tokenizer.push_to_hub("distilbert_ORO_Branch", use_auth_token = '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')