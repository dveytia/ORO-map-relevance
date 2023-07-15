
def KFoldRandom(n_splits, X, no_test, shuffle=False, discard=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    for train, test in kf.split(X):
        if not discard:
            train = list(train) +  [x for x in test if x in no_test]
        test = [x for x in test if x not in no_test]
        yield (train, test)

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


def evaluate_preds(y_true, y_pred, targets):
    res = {}
    for average in ["micro","macro","weighted", "samples"]:
        try:
            res[f'ROC AUC {average}'] = roc_auc_score(y_true, y_pred, average=average)
        except:
            res[f'ROC AUC {average}'] = np.NaN
        res[f'F1 {average}'] = f1_score(y_true, y_pred.round(), average=average)
        res[f'precision {average}'] = precision_score(y_true, y_pred.round(), average=average)
        res[f'recall {average}'] = recall_score(y_true, y_pred.round(), average=average)
        
    for i, target in enumerate(targets):
        try:
            res[f'ROC AUC - {target}'] = roc_auc_score(y_true[:,i], y_pred[:,i])
        except:
            res[f'ROC AUC - {target}'] = np.NaN
        res[f'precision - {target}'] = precision_score(y_true[:,i], y_pred[:,i].round())
        res[f'recall - {target}'] = recall_score(y_true[:,i], y_pred[:,i].round())
        res[f'F1 - {target}'] = f1_score(y_true[:,i], y_pred[:,i].round())
        res[f'accuracy - {target}'] = accuracy_score(y_true[:,i], y_pred[:,i].round())
        res[f'n_target - {target}'] = y_true[:,i].sum()
    
    return res



def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
            

            
def train_eval_bert(params, df, train, test):
    train_dataset, val_dataset, MAX_LEN = create_train_val(df['text'], df['labels'], train, test)
    
    print("training bert with these params")
    print(params)
    model = init_model('distilbert-base-uncased', len(targets), params)
    model.fit(train_dataset.shuffle(100).batch(params['batch_size']),
              epochs=params['num_epochs'],
              batch_size=params['batch_size'],
              class_weight=params['class_weight']
    )

    preds = model.predict(val_dataset.batch(1)).logits
    y_pred = tf.keras.activations.sigmoid(tf.convert_to_tensor(preds)).numpy()
#    ai = np.expand_dims(np.argmax(y_pred, axis=1), axis=1)
#    maximums = np.maximum(y_pred.max(1),0.51)
#    np.put_along_axis(y_pred, ai, maximums.reshape(ai.shape), axis=1)
    eps = evaluate_preds(np.array(df.loc[test,targets]), y_pred, targets)
    print(eps)
    for key, value in params.items():
        eps[key] = value
    return eps