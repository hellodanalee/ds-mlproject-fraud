
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns



# Calculate (new_index - old_index) 
# Create a total of all consumption levels 
# Create aggregation features for consumption and index
def create_consump_agg(feature_dataframe, invoice) :

    df_copy = invoice.copy()
    
    df_copy['f_index_diff'] = invoice['new_index'] - invoice['old_index']
    
    df_copy['f_total_consumption'] = (
        df_copy['consommation_level_1'].fillna(0) +
        df_copy['consommation_level_2'].fillna(0) +
        df_copy['consommation_level_3'].fillna(0) +
        df_copy['consommation_level_4'].fillna(0)
    )

    analysis_cols = ['consommation_level_1', 'consommation_level_2', 
                'consommation_level_3', 'consommation_level_4', 'f_index_diff', 'f_total_consumption']
    
    
    aggregated_df = df_copy.groupby('client_id')[analysis_cols].agg(['min', 'max', 'std', 'mean'])

    # change the column names 
    aggregated_df.columns = ['_'.join(col) for col in aggregated_df.columns]
    
    # reset the index 
    aggregated_df = aggregated_df.reset_index()
    
    # merge to the feature_dataframe
    feature_dataframe = feature_dataframe.merge(aggregated_df, on='client_id', how='left')

    return feature_dataframe

# Aggregate Tarif 
def create_tarif_agg(feature_dataframe, invoice):

    df_copy = invoice[['client_id', 'tarif_type']].copy()
    
    aggregated = df_copy.groupby('client_id')['tarif_type'].agg(
        lambda x: pd.Series.mode(x)[0]
    ).reset_index()
    

    print(aggregated.head())    
    feature_dataframe = feature_dataframe.merge(aggregated, on='client_id', how='left')
    
    return  feature_dataframe


# Calculate a Mutual Information score 
def calculate_mutual_information(feature_dataframe, target_col='target', exclude_cols=None, random_state=50):

    if exclude_cols is None:
        exclude_cols = []
    
    exclude_cols = exclude_cols + [target_col]
    feature_columns = [col for col in feature_dataframe.columns if col not in exclude_cols]
    
    # Assign X, y values
    X = feature_dataframe[feature_columns].fillna(0)  # Convert missing values to 0
    y = feature_dataframe[target_col]
    
    mi_scores = mutual_info_classif(X, y, random_state=random_state) # How can i import the class?
    
    mi_results = pd.DataFrame({
        'Feature': feature_columns,
        'MI_Score': mi_scores
    })
    
    # Order the result desc
    mi_results = mi_results.sort_values('MI_Score', ascending=False).reset_index(drop=True)
    
    return mi_results



# Visualize the MI results
def visualize_mutual_information(mi_results, title='MI on each feature', 
                                 xlabel='Mutual Information', ylabel='Feature',
                                 figsize=(12, 10), top_n=None, show=True):
    # Set a Top_n 
    if top_n is not None and top_n < len(mi_results):
        plot_data = mi_results.head(top_n).copy()
    else:
        plot_data = mi_results.copy()
    
    # Plot a graph 
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='MI_Score', y='Feature', data=plot_data)
    
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Show a value index (선택적)
    for i, v in enumerate(plot_data['MI_Score']):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return plt.gcf()




if __name__ == "__main__":
    invoice_df = pd.read_csv('data/invoice_train.csv')
    client_df = pd.read_csv('data/client_train.csv')    

    feature_dataframe = client_df[['client_id']].copy()
    
    feature_dataframe =create_tarif_agg(feature_dataframe, invoice_df )

    print(feature_dataframe['tarif_type'].unique())