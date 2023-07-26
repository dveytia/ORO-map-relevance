import pandas as pd


# Check model scores for each run

def get_best_model_single(file_path_prefix, k_range):
    inner_scores = []
    params = ['batch_size','weight_decay','learning_rate','num_epochs','class_weight']
    
    for k in range(k_range): #Change to 5 if you are using the binary
        inner_df = pd.read_csv(f'{file_path_prefix}{k}.csv')
        inner_df = inner_df.sort_values('F1',ascending=False).reset_index(drop=True)
        inner_scores += inner_df.to_dict('records')
    
    inner_scores = pd.DataFrame.from_dict(inner_scores).fillna(-1)
    best_model = (inner_scores
                  .groupby(params).agg({
                      'F1':'mean',
                      'ROC AUC':'mean',
                      'precision':'mean',
                      'recall':'mean',
                      'accuracy':'mean'
                      }).sort_values('F1',ascending=False).reset_index()).to_dict('records')[0]
    
    del inner_scores, inner_df
    return best_model




def get_best_model_multi(file_path_prefix, k_range):
    inner_scores = []
    params = ['batch_size','weight_decay','learning_rate','num_epochs','class_weight']
    
    for k in range(k_range): 
        inner_df = pd.read_csv(f'{file_path_prefix}{k}.csv')
        inner_df = inner_df.sort_values('F1 macro', ascending=False).reset_index(drop=True)
        inner_scores += inner_df.to_dict('records')
    
    inner_scores = pd.DataFrame.from_dict(inner_scores).fillna(-1)
    
    if 'accuracy macro' not in list(inner_scores.columns): # if there is no accuracy macro column set to dummy value
        inner_scores['accuracy macro'] = -999 
        
    best_model = (inner_scores
                  .groupby(params).agg({
                      'F1 macro':'mean',
                      'ROC AUC macro':'mean',
                      'precision macro':'mean',
                      'recall macro':'mean',
                      'accuracy macro':'mean'
                      }).sort_values('F1 macro',ascending=False).reset_index()).to_dict('records')[0]
    
    
    best_model.rename(columns={'F1 macro': 'F1', 'ROC AUC macro': 'ROC AUC', 'precision macro':'precision', 'recall macro':'recall', 'accuracy macro':'accuracy'}, inplace=True)
    
    del inner_scores, inner_df
    return best_model


# test
file_path_prefix = f'/home/dveytia/ORO-map-relevance/outputs/model_selection/adapt_to_threat_model_selection_'
inner_scores = []
params = ['batch_size','weight_decay','learning_rate','num_epochs','class_weight']
k=0
inner_df = pd.read_csv(f'{file_path_prefix}{k}.csv')
inner_df = inner_df.sort_values('F1',ascending=False).reset_index(drop=True)
inner_scores += inner_df.to_dict('records')
    
inner_scores = pd.DataFrame.from_dict(inner_scores).fillna(-1)
best_model = (inner_scores
                  .groupby(params).agg({
                      'F1':'mean',
                      'ROC AUC':'mean',
                      'precision':'mean',
                      'recall':'mean',
                      'accuracy':'mean'
                      }).sort_values('F1',ascending=False).reset_index()).to_dict('records')[0]
    

best_model


#These are PARTIAL paths (the end number gets added on in the function) to the model selection of each corresponding run
targetVar = "adapt_to_threat"

test1=get_best_model(f'/home/dveytia/ORO-map-relevance/outputs/model_selection/adapt_to_threat_model_selection_', 3)
test2=get_best_model(f'/home/dveytia/ORO-map-relevance/outputs/model_selection/climate_mitigation_model_selection_', 3)


nokey=get_best_model(r"C:\Users\vcm20gly\OneDrive - Bangor University\Documents\Review\02_Binary-NoKeywords\model_selection_nokey\model_selection_")
discard_true=get_best_model(r"C:\Users\vcm20gly\OneDrive - Bangor University\Documents\Review\02_Binary-discardTrue\model_selection_discardTrue\model_selection_")
mitigation=get_best_model(r"C:\Users\vcm20gly\OneDrive - Bangor University\Documents\Review\02_Binary-OROBranch-Mitigation\model_selection_mitigation\model_selection_mitigation_")
nature=get_best_model(r"C:\Users\vcm20gly\OneDrive - Bangor University\Documents\Review\02_Binary-OROBranch-Natural\model_selection_nature\model_selection_nature_")
societal=get_best_model(r"C:\Users\vcm20gly\OneDrive - Bangor University\Documents\Review\02_Binary-OROBranch-Societal\model_selection_societal\model_selection_societal_")
unclear=get_best_model(r"C:\Users\vcm20gly\OneDrive - Bangor University\Documents\Review\02_Binary-OROBranch-Unclear\model_selection_unclear\model_selection_unclear_")
exclusion=get_best_model(r"C:\Users\vcm20gly\OneDrive - Bangor University\Documents\Review\03_Binary-AllText-NewApproach\model_selection_excl\model_selection_")
run3alltext=get_best_model(r"C:\Users\vcm20gly\OneDrive - Bangor University\Documents\Review\03_Binary-AllText\model_selection_3\model_selection_")

#Adds them all together
best_models_all=pd.DataFrame([all_text, nokey, discard_true, mitigation, nature, societal, unclear, exclusion, run3alltext])
best_models_all["run"]=['all_text', 'nokey', 'discard_true', 'mitigation', 'nature', 'societal', 'unclear', 'exclusion', 'run3alltext']

