import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################# Change INPUTS ##################
binVar = "societal_implemented" # name of binary variable
binVarFull = "oro_development_stage.Implemented_continued_assessment"
dockerFilePath = '/home/devi/analysis/'

conditionVar = 'oro_branch'
conditionVarVal = 'oro_branch.Societal'
codedVariablesTxt = dockerFilePath + 'data/seen/all-coding-format-distilBERT-simplifiedMore.txt'
screenDecisionsTxt = dockerFilePath + 'data/seen/all-screen-results_screenExcl-codeIncl.txt'
unseenTxt = dockerFilePath + 'data/unseen/0_unique_references.txt' # change to unique_references2.txt?
relevanceTxt = dockerFilePath + 'outputs/predictions-compiled/1_document_relevance_13062023.csv'
n_threads = 2 # number of threads to parallelize on


################# Load documents and format ##################

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
cond_df = pd.read_csv(f'{dockerFilePath}outputs/predictions-compiled/{conditionVar}_predictions.csv')

# Merge all unseen dataframes with their predictions
unseen_df = unseen_df.merge(pred_df, how="left")
unseen_df = unseen_df.merge(cond_df, how="left")
unseen_df['seen']=0


# Choose which predictiction boundaries to apply
unseen_df = unseen_df[unseen_df['0 - relevance - upper_pred']>=0.5] # has to first be relevant
unseen_df = unseen_df[unseen_df[(conditionVarVal + ' - upper_pred')]>=0.5] # has to then be relevant for conditional variable


# Concatenate seen and unseen
df = (pd.concat([seen_df,unseen_df])
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)

df['text'] = df['title'].astype("str") + ". " + df['abstract'].astype("str") + " " + "Keywords: " + df["keywords"].astype("str")
df['text'] = df.apply(lambda row: (row['title'] + ". " + row['abstract']) if pd.isna(row['text']) else row['text'], axis=1)

#################### Rename target column #################################
df = df.rename(columns={binVarFull: binVar})

seen_index = df[df['seen']==1].index
unseen_index = df[df['seen']==0].index

print("Dataset has been re-formatted and is ready")


################# using unseen_ids file to compile preds #####################
unseen_ids= pd.DataFrame(np.load(f"{dockerFilePath}outputs-docker/predictions_data/{binVar}_unseen_ids.npz")) #Change file path
unseen_ids.columns=["id"]

k = 5 #Unless you rerun the initial binary predictions you don't need to change this.

if k==10:
    y_preds = np.zeros((len(unseen_ids),10))

    for k in range(10):
        #y_pred = np.load(rf'home\dveytia\ORO-map-relevance\outputs\predictions\{binVar}_y_preds_5fold_{k}.npz.npy')[:,0]#Change file path
        y_pred = np.load(f'{dockerFilePath}outputs-docker/predictions/{binVar}_y_preds_10fold_{k}.npz.npy')[:,0]# OR this?
        y_preds[:,k] = y_pred
        print(np.where(y_pred>0.5,1,0).sum())    
else:
    y_preds = np.zeros((len(unseen_ids),5))

    for k in range(5):
        y_pred = np.load(f'{dockerFilePath}outputs-docker/predictions/{binVar}_y_preds_5fold_{k}.npz.npy')[:,0]#Change file path
        y_preds[:,k] = y_pred
        print(np.where(y_pred>0.5,1,0).sum())
    
mean_pred = np.mean(y_preds, axis=1)
std_pred = np.std(y_preds, axis=1)

preds_upper = np.minimum(mean_pred + std_pred, 1)
preds_lower = np.maximum(mean_pred - std_pred, 0)

unseen_ids['0 - relevance - mean_prediction'] = mean_pred
unseen_ids['0 - relevance - std_prediction'] = std_pred
unseen_ids['0 - relevance - lower_pred'] = preds_lower
unseen_ids['0 - relevance - upper_pred'] = preds_upper

           

unseen_ids.to_csv(f'{dockerFilePath}outputs-docker/predictions-compiled/{binVar}_predictions.csv',index=False) # Save file, change path


