import pandas as pd
import joblib
from lightgbm import LGBMRegressor


def train_test(data, features, stores, d_store_id, folder_name, model_name):
    ''' Train and Test data with a list of features'''
    # DataFrame with filter scope
    data_scope = data[features].copy()
    # Validation Set
    valid = data_scope[(data_scope['d']>=1914) & (data_scope['d']<1942)][['id','d','sold']]
    # Validation Prediction
    valid_set = valid['sold']
    # Validation + Predicition for all stores by step 
    df_validpred = pd.DataFrame()

    # Loop for training a model for each store
    for store in stores:
        # Dataframe for each store
        df = data_scope[data_scope['store_id']==store]
        # Train, Valid and Test sets
        df_train = df[df['d']<1914].copy()
        df_valid = df[(df['d']>=1914) & (df['d']<1942)].copy()
        df_test = df[df['d']>=1942].copy()
        # Train Data until day = 1914
        X_train, y_train = df_train.drop('sold',axis=1), df_train['sold']
        # Validation Day: 1914 to 1942
        X_valid, y_valid = df_valid.drop('sold',axis=1), df_valid['sold']
        # X_test with 
        X_test = df_test.drop('sold',axis=1)

        # LGBM Model
        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=8,
            num_leaves=50,
            min_child_weight=300
        )

        # Fit model
        model.fit(X_train, y_train, 
                eval_set = [(X_train,y_train),(X_valid,y_valid)], 
                eval_metric = 'rmse', 
                verbose = 0, 
                early_stopping_rounds = 20)
        # Compute Prediction
        valid_pred = model.predict(X_valid)
        # Actual Validation vs. Prediction
        df_valid = pd.DataFrame({
            'validation':valid_set[X_valid.index],
            'prediction':valid_pred,
            'store': d_store_id[store]
        })
        df_valid['error'] = df_valid['validation'] - df_valid['prediction']
        df_validpred = pd.concat([df_validpred, df_valid])
        # Save model
        filename = folder_name + 'model-{}-'.format(model_name) + str(d_store_id[store])+'.pkl'
        joblib.dump(model, filename)
        del model, X_train, y_train, X_valid, y_valid

    # Save Prediction for all stores
    df_validpred.to_csv(folder_name + 'error_prediction-{}.csv'.format(model_name))

    return df_validpred