import pandas as pd
import numpy as np


################# Change INPUTS ##################
dockerFilePath = '/home/devi/analysis/'

targetVar = "method_type" # name of variable 
suffix = "simplified" # the suffix to add to this run of the variable 
conditionVar = "data" # the variable(s) that has to ==1 in order to predict the target Var
conditionVarVal = "data_type.Primary"
dropLabels = ['method_type.Expert_opinion', 'method_type.Method_or_technology_development']
mergeLabels= ['method_type.Experimental_exsitu', 'method_type.Experimental_insitu', 'method_type.Observational'] # list of labels to merge
newLabel = 'method_type.Empirical' # name of new label that encompasses the merged labels

codedVariablesTxt = dockerFilePath + 'data/seen/all-coding-format-distilBERT-simplifiedMore.txt'
screenDecisionsTxt = dockerFilePath + 'data/seen/all-screen-results_screenExcl-codeIncl.txt'
unseenTxt = dockerFilePath + 'data/unseen/0_unique_references.txt' # change to unique_references2.txt?
relevanceTxt = dockerFilePath + 'outputs/predictions-compiled/1_document_relevance_13062023.csv'

################# Load documents and format ##################

# Load seen documents
seen_df = pd.read_csv(codedVariablesTxt, delimiter='\t') 
seen_df = seen_df.rename(columns={'analysis_id':'id'})
seen_df['seen']=1

# Load unseen documents and merge
unseen_df = pd.read_csv(unseenTxt, delimiter='\t') 
unseen_df = unseen_df.rename(columns={'analysis_id':'id'})
unseen_df=unseen_df.dropna(subset=['abstract']).reset_index(drop=True)
unseen_df['seen']=0

# Load prediction relevance and merge
pred_df = pd.read_csv(relevanceTxt) 
unseen_df = unseen_df.merge(pred_df, how="left")

# Load predicted relevance of conditional variable and merge
cond_df = pd.read_csv(dockerFilePath + 'outputs/predictions-compiled/'+ conditionVar +'_predictions.csv')
unseen_df = unseen_df.merge(cond_df, how="left")

# Choose which predictiction boundaries to apply
unseen_df = unseen_df[unseen_df['0 - relevance - upper_pred']>=0.5] # has to be relevant overall
unseen_df = unseen_df[unseen_df[(conditionVarVal + ' - upper_pred')]>=0.5] # has to then be relevant for conditional variable

# Concatenate seen and unseen
df = (pd.concat([seen_df,unseen_df])
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)

df['text'] = df['title'].astype("str") + ". " + df['abstract'].astype("str") + " " + "Keywords: " + df["keywords"].astype("str")
df['text'] = df.apply(lambda row: (row['title'] + ". " + row['abstract']) if pd.isna(row['text']) else row['text'], axis=1)


# Make an index of which articles are seen or unseen
seen_index = df[df['seen']==1].index
unseen_index = df[df['seen']==0].index




################# Merge variables to simplify labels of predicted variable ###############
# Merge into one column
df[newLabel] = df[mergeLabels].sum(axis = 'columns') # get the sum across all the columns to merge
df[newLabel].where(df[newLabel] <= 1, 1) # cap value at 1

# Remove old columns
df = df.drop(columns = mergeLabels)
df = df.drop(columns = dropLabels)

print("The data has been re-formatted")
print(df.shape)



################# using unseen_ids file to compile preds #####################
unseen_ids= pd.DataFrame(np.load(f'{dockerFilePath}outputs-docker/predictions_data/{targetVar}_{suffix}_data_pred_ids.npy')) #Change file path

unseen_ids.columns=["id"]

targets = [x for x in df.columns if targetVar in x]

y_preds = [ np.zeros((len(unseen_ids),5)) for x in range(len(targets))]

all_cols = ['id']

for k in range(5):
    y_pred = np.load(f"{dockerFilePath}outputs-docker/predictions/{targetVar}_{suffix}_y_preds_5fold_data_{k}.npz.npy") #Load results, change file path
    
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
    

unseen_ids.to_csv(f'{dockerFilePath}outputs-docker/predictions-compiled/{targetVar}_{suffix}_predictions.csv',index=False) #Saves .csv file, change file path


