import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################# Change INPUTS ##################
targetVar = "method_type" # name of variable
conditionVar = "data"
conditionVarVal = "data_type.Primary" # the variable that has to ==1 in order to predict the target Var
suffix = "nested" # the suffix to add to this run of the variable 
codedVariablesTxt = '/home/dveytia/ORO-map-relevance/data/seen/all-coding-format-distilBERT-simplifiedMore.txt'
screenDecisionsTxt = '/home/dveytia/ORO-map-relevance/data/seen/all-screen-results_screenExcl-codeIncl.txt'
unseenTxt = '/home/dveytia/ORO-map-relevance/data/unseen/0_unique_references.txt' # change to unique_references2.txt?
relevanceTxt = '/home/dveytia/ORO-map-relevance/outputs/predictions-compiled/1_document_relevance_13062023.csv'


################# Load documents and format ##################

# Load seen documents
seen_df = pd.read_csv(codedVariablesTxt, delimiter='\t')
seen_df = seen_df.rename(columns={'analysis_id':'id'})
seen_df['seen']=1

# Load unseen documents 
unseen_df = pd.read_csv(unseenTxt, delimiter='\t')
unseen_df = unseen_df.rename(columns={'analysis_id':'id'})
unseen_df=unseen_df.dropna(subset=['abstract']).reset_index(drop=True)

# Load prediction relevance
pred_df = pd.read_csv(relevanceTxt) 
cond_df = pd.read_csv(f'/home/dveytia/ORO-map-relevance/outputs/predictions-compiled/{conditionVar}_predictions.csv')

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


seen_index = df[df['seen']==1].index
unseen_index = df[df['seen']==0].index

print("Dataset has been re-formatted and is ready")


################# using unseen_ids file to compile preds #####################
unseen_ids= pd.DataFrame(np.load(f'/home/dveytia/ORO-map-relevance/outputs/predictions_data/{targetVar}_{suffix}_data_pred_ids.npy')) #Change file path
unseen_ids.columns=["id"]

targets = [x for x in df.columns if targetVar in x]

y_preds = [ np.zeros((len(unseen_ids),5)) for x in range(len(targets))]

all_cols = ['id']

for k in range(5):
    y_pred = np.load(f"/home/dveytia/ORO-map-relevance/outputs/predictions/{targetVar}_{suffix}_y_preds_5fold_data_{k}.npz.npy") #Load results, change file path
    
    for i in range(len(targets)):
        y_preds[i][:,k] = y_pred[:,i]
        
for i in range(len(targets)):
    mean_pred = np.mean(y_preds[i], axis=1)
    std_pred = np.std(y_preds[i], axis=1)

    preds_upper = np.minimum(mean_pred + std_pred, 1)
    preds_lower = np.maximum(mean_pred - std_pred, 0)
    
    t = targets[i]
    
    unseen_ids[f'{t} - mean_prediction'] = mean_pred
    unseen_ids[f'{t} - std_prediction'] = std_pred
    unseen_ids[f'{t} - lower_pred'] = preds_lower
    unseen_ids[f'{t} - upper_pred'] = preds_upper
    
    print(targets[i]) 
    print(unseen_ids.sort_values(f'{t} - mean_prediction',ascending=False).head())
    

unseen_ids.to_csv(f'/home/dveytia/ORO-map-relevance/outputs/predictions-compiled/{targetVar}_{suffix}_predictions.csv',index=False) #Saves .csv file, change file path


