# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:05:00 2023

@author: vcm20gly

Compile predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################# Change INPUTS ##################
binVar = "Forecast" # name of binary variable to fit
codedVariablesTxt = '/home/dveytia/ORO-map-relevance/data/seen/all-coding-format-distilBERT-simplifiedMore.txt'
screenDecisionsTxt = '/home/dveytia/ORO-map-relevance/data/seen/all-screen-results_screenExcl-codeIncl.txt'
unseenTxt = '/home/dveytia/ORO-map-relevance/data/unseen/0_unique_references.txt' # change to unique_references2.txt?
relevanceTxt = '/home/dveytia/ORO-map-relevance/outputs/predictions-compiled/1_document_relevance_13062023.csv'




######## Load files, change paths #################
# load seen documents
seen_df = pd.read_csv(codedVariablesTxt, delimiter='\t')
seen_df = seen_df.rename(columns={'analysis_id':'id'})
seen_df['seen']=1

# Load unseen documents and merge
unseen_df = pd.read_csv(unseenTxt, delimiter='\t') 
unseen_df = unseen_df.rename(columns={'analysis_id':'id'})
unseen_df=unseen_df.dropna(subset=['abstract']).reset_index(drop=True)

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

################# using unseen_ids file to compile preds #####################
unseen_ids= pd.DataFrame(np.load(f'/home/dveytia/ORO-map-relevance/outputs/predictions_data/{binVar}_unseen_ids.npz.npy'))
unseen_ids.columns=["id"]

k = 5 #Unless you rerun the initial binary predictions you don't need to change this.

if k==10:
    y_preds = np.zeros((len(unseen_ids),10))

    for k in range(10):
        #y_pred = np.load(rf'home\dveytia\ORO-map-relevance\outputs\predictions\{binVar}_y_preds_5fold_{k}.npz.npy')[:,0]#Change file path
        y_pred = np.load(f'/home/dveytia/ORO-map-relevance/outputs/predictions/{binVar}_y_preds_10fold_{k}.npz.npy')[:,0]# OR this?
        y_preds[:,k] = y_pred
        print(np.where(y_pred>0.5,1,0).sum())    
else:
    y_preds = np.zeros((len(unseen_ids),5))

    for k in range(5):
        y_pred = np.load(f'/home/dveytia/ORO-map-relevance/outputs/predictions/{binVar}_y_preds_5fold_{k}.npz.npy')[:,0]#Change file path
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

           

unseen_ids.to_csv(f'/home/dveytia/ORO-map-relevance/outputs/predictions-compiled/{binVar}_predictions.csv',index=False) # Save file, change path



#################### Create figure for inclusions ####################
fig, ax = plt.subplots(dpi=150)

b = np.mean(y_preds, axis = 1)
idx = b.argsort()
y_preds_sorted = np.take(y_preds, idx, axis=0)

mean_pred = np.mean(y_preds_sorted, axis=1)
std_pred = np.std(y_preds_sorted, axis=1)

ax.plot(mean_pred, color='r', label="Mean")

preds_upper = np.minimum(mean_pred + std_pred, 1)
preds_lower = np.maximum(mean_pred - std_pred, 0)

ax.fill_between(range(len(mean_pred)), preds_upper, preds_lower, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

lb = preds_upper[np.where(preds_upper>0.5)].shape[0]
ub = preds_lower[np.where(preds_lower>0.5)].shape[0]
mb = mean_pred[np.where(mean_pred>0.5)].shape[0]

s = f'{mb:,} ({ub:,}-{lb:,})\n relevant documents predicted'

ax.plot([np.argwhere(preds_upper>0.5)[0][0]*0.75,np.argwhere(preds_upper>0.5)[0][0]],[0.6,0.5],c="grey",ls="--")
ax.plot([np.argwhere(preds_upper>0.5)[0][0]*0.75,np.argwhere((preds_lower>0.5) & (preds_lower < 0.501))[-1][0]],[0.6,0.5],c="grey",ls="--")
ax.text(np.argwhere(preds_upper>0.5)[0][0]*0.75,0.6,s,ha="right",va="bottom",bbox=props)

ax.set_xlabel('Documents')
ax.set_ylabel('Predicted relevance')

ax.legend()
plt.savefig(f'/home/dveytia/ORO-map-relevance/figures/{binVar}_predictions_unseen.png',bbox_inches="tight") # Save plot, change file path and name


