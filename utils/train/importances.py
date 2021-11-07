
import pandas as pd 
import seaborn as sns 
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path



def features_importance(folder_model, model_name, new_features, d_store_id, valid_rmse, stores, n_features):
    ''' Plot features importance of the model'''
    Path(folder_model).mkdir(parents=True, exist_ok=True)

    # Dataframe for features importances
    feature_importance_df = pd.DataFrame()
    features = [f for f in new_features if f != 'sold']

    for store in stores:
        store_name = d_store_id[store]
        filename = folder_model + 'model-{}-'.format(model_name) + str(d_store_id[store])+'.pkl'
        # load model
        model = joblib.load(filename)

        # Create features importance for this model
        store_importance_df = pd.DataFrame()
        store_importance_df["feature"] = features
        store_importance_df["importance"] = model.feature_importances_
        store_importance_df["store"] = store_name

        # Concat
        feature_importance_df = pd.concat([feature_importance_df, store_importance_df], axis=0)

    # Features Importances Analysis
    df_fi_mean = pd.DataFrame(feature_importance_df[
        ["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False))
    df_fi_mean.columns = ['importance']
    df_fi_mean['%_importance'] = (100 * df_fi_mean['importance']/df_fi_mean['importance'].sum(axis =0)).round(2)
    # df_fi_mean.to_excel(folder_model + 'features_importance.xlsx')


    # Plot
    cols = df_fi_mean[:n_features].index
    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
    fig = plt.figure(figsize=(6,10))
    ax = fig.gca()
    sns.barplot(x="importance", y="feature", data = best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Top {} Features'.format(n_features))
    plt.tight_layout()
    print("{}: RMSE = {}".format('filtered', valid_rmse))
    fig.savefig(folder_model + 'features_importance.png')
    plt.show()