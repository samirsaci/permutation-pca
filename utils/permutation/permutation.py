import pandas as pd
import joblib
import numpy as np



def permutation(data, all_features, stores, d_store_id, baseline_rmse, dict_error, FOLDER_MODEL):
    ''' Simulate the impact on RMSE if we remove each feature once at a time'''
    
    # All validation predition
    df_validall = pd.DataFrame()

    # Validation Set
    valid = data[(data['d']>=1914) & (data['d']<1942)][['id','d','sold']]
    # Validation Prediction
    valid_set = valid['sold']

    for col in all_features:
        # Validation + Predicition for all stores by step 
        df_validpred = pd.DataFrame()

        # DataFrame with filter scope
        data_scope = data[all_features].copy()

        # Loop for training a model for each store
        for store in stores:

            # Dataframe for each store
            df = data_scope[data_scope['store_id']==store]

            # Train, Valid and Test
            df_valid = df[(df['d']>=1914) & (df['d']<1942)].copy()
            df_test = df[df['d']>=1942].copy()
            
            # Shuffle columns for validation test
            if df_valid[col].dtypes.name != 'category':
                df_valid[col] = np.random.permutation(df_valid[col].values)

            # Validation Day: 1914 to 1942
            X_valid, y_valid = df_valid.drop('sold',axis=1), df_valid['sold']

            # X_test with 
            X_test = df_test.drop('sold',axis=1)

            # Load Model
            store_name = d_store_id[store]
            # load model
            filename = FOLDER_MODEL + 'model-initial-' + str(d_store_id[store])+'.pkl'
            model = joblib.load(filename)

            # Compute Prediction
            valid_pred = model.predict(X_valid)

            # Actual Validation vs. Prediction
            df_valid = pd.DataFrame({
                'validation':valid_set[X_valid.index],
                'prediction':valid_pred,
                'store': d_store_id[store],
            })
            df_valid['error'] = df_valid['validation'] - df_valid['prediction']
            df_valid['features_removed'] = col
            df_validpred = pd.concat([df_validpred, df_valid])

            del model, X_valid, y_valid, X_test

        # Compute Error
        valid_rmse = 100 * np.sqrt(np.mean((df_validpred.validation.values - df_validpred.prediction.values) ** 2))/np.mean(df_validpred.validation.values)
        # Add Error in a Dictionnary
        dict_error[col] = valid_rmse
        print("Remove {}: delta_rmse = {}".format(col, valid_rmse - baseline_rmse))
        
        # Add results
        df_validall = pd.concat([df_validall, df_validpred])
        
    # Final results
    df_permres = pd.DataFrame({'remove':all_features,
                'baseline':[baseline_rmse for col in all_features],
                'new_rmse':[dict_error[col] for col in all_features]})
    df_permres['delta'] = df_permres['new_rmse'] - df_permres['baseline']
    
    return df_permres, df_validall, dict_error
