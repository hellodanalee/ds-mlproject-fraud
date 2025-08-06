"""
w1_feature_fraud_mk.py

Utility objects for the fraud‑detection project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Union

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Fraud:
    def __init__(
        self,
        csv_files: Union[Iterable[str], Iterable[Path]],
        target_column: str | None = None,
    ) -> None:
        self._frames: Dict[str, pd.DataFrame] = {}

        for path in csv_files:
            p = Path(path).expanduser().resolve()

            # Skip duplicate entries
            if str(p) in self._frames:
                continue

            try:
                self._frames[str(p)] = pd.read_csv(p)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"CSV file not found: {p}") from exc
            except pd.errors.EmptyDataError:
                self._frames[str(p)] = pd.DataFrame()  # empty placeholder

        # Optional: extract target series if requested
        self._target: pd.Series | None = None
        if target_column is not None:
            # find first DataFrame containing the target column
            for df in self._frames.values():
                if target_column in df.columns:
                    self._target = df.set_index("client_id")[target_column]
                    break
            if self._target is None:
                print(f"Warning: Target column '{target_column}' not found in any loaded DataFrame. Proceeding without target.")

    # ------------------------------------------------------------------ #
    # Dictionary‑style helpers
    # ------------------------------------------------------------------ #
    def __getitem__(self, key: Union[str, Path]) -> pd.DataFrame:
        return self._frames[str(Path(key).expanduser().resolve())]

    def __iter__(self):
        return iter(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def keys(self) -> List[str]:
        """Return a list of absolute file paths contained in the object."""
        return list(self._frames.keys())

    def items(self):
        return self._frames.items()

    def values(self):
        return self._frames.values()

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    def get(self, key: Union[str, Path], default=None) -> pd.DataFrame:
        """dict‑style ``get`` method."""
        return self._frames.get(str(Path(key).expanduser().resolve()), default)


    def to_dict(self) -> Dict[str, pd.DataFrame]:
        """Return the internal mapping as a plain dict."""
        return dict(self._frames)

    def get_target(self, client) -> pd.DataFrame:
        # Basis-Spalten, die immer zurückkommen sollen
        columns = ["client_id", "disrict", "region", "client_catg"]

        # Nur hinzufügen, wenn target wirklich da ist
        if "target" in client.columns:
            columns.insert(1, "target")
        else:
            print("Warning: 'target' column not found. Returning features without target.")

        # Nochmal absichern, dass alle Spalten existieren
        missing = [c for c in columns if c not in client.columns]
        if missing:
            raise KeyError(f"Columns not found in client DataFrame: {missing}")

        return client[columns].copy()

# ---------------------------------------------------------------------- #
# Data‑frame utilities
# ---------------------------------------------------------------------- #
def left_join_on(
    key: str,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    suffixes: tuple[str, str] = ("_left", "_right"),
    validate: str | None = "one_to_many",
) -> pd.DataFrame:
    if key not in left_df.columns:
        raise KeyError(f"'{key}' not found in left_df columns")
    if key not in right_df.columns:
        raise KeyError(f"'{key}' not found in right_df columns")

    merged = pd.merge(
        left_df,
        right_df,
        how="left",
        on=key
    )
    return merged   

def add_invoice_frequency_features(
    merge_df: pd.DataFrame, collection_df: pd.DataFrame
) -> pd.DataFrame:
    # Copy to avoid modifying original
    df_copy = merge_df.copy()
    # Convert invoice_date to datetime
    df_copy["invoice_date_as_datetime"] = pd.to_datetime(
        df_copy["invoice_date"], format="%Y-%m-%d", errors="coerce"
    )
    # Sort by client and date for correct diff calculation
    df_copy = df_copy.sort_values(["client_id", "invoice_date_as_datetime"])
    # Compute days between consecutive invoices per client
    df_copy["delta_days"] = (
        df_copy.groupby("client_id")["invoice_date_as_datetime"]
               .diff()
               .dt.days
    )
    # Aggregate median days per client
    median_days = df_copy.groupby("client_id")["delta_days"].median()
    # Convert to months and years
    median_months = median_days / 30.44
    median_years = median_days / 365.25
    # Build frequency frame
    freq_df = pd.DataFrame({
        "client_id": median_days.index,
        "f_invoive_date_diff_days": median_days.values,
        "f_invoive_date_median_months": median_months.values,
        "f_invoive_date_median_years": median_years.values,
    })
    # Merge aggregated features back into original DataFrame
    return collection_df.merge(freq_df, on="client_id", how="left")


def add_counter_statue_error_occured_features(
    merge_df: pd.DataFrame, collection_df: pd.DataFrame
) -> pd.DataFrame:
    # Copy to avoid modifying original
    df_copy = merge_df.copy()
    # Determine if any error occurred per client (status != '0')
    error_series = (
        df_copy
        .groupby("client_id")["counter_statue"]
        .apply(lambda x: int(x.astype(str).ne("0").any()))
        .rename("f_counter_statue_error_occured")
    )
    # Merge the feature back into collection_df
    return collection_df.merge(error_series.reset_index(), on="client_id", how="left")


# ---------------------------------------------------------------------- #
# New feature: add_counter_regions_features
# ---------------------------------------------------------------------- #
def add_counter_regions_features(
    merge_df: pd.DataFrame, collection_df: pd.DataFrame
) -> pd.DataFrame:
    # Copy to avoid modifying original
    df_copy = merge_df.copy()
    # Count unique regions per client
    region_counts = (
        df_copy
        .groupby("client_id")["region"]
        .nunique()
        .gt(1)
        .astype(int)
        .rename("f_counter_regions")
    )
    # Merge the feature back into collection_df
    return collection_df.merge(region_counts.reset_index(), on="client_id", how="left")


# ---------------------------------------------------------------------- #
# New feature: add_region_fraud_rate_features
# ---------------------------------------------------------------------- #
def add_region_fraud_rate_features(
    merge_df: pd.DataFrame, collection_df: pd.DataFrame
) -> pd.DataFrame:
    # 1) Fraud-Rate pro Region berechnen
    region_rates = (
        merge_df
        .groupby("region")["target"]
        .mean()  # Anteil target==1
        .rename("f_t_region_fraud_rate")
        .reset_index()
    )

    # 2) Jedem Client seine Region-Rate zuordnen
    client_region = merge_df[["client_id", "region"]].drop_duplicates()
    client_rate = client_region.merge(
        region_rates, on="region", how="left"
    )

    # 3) In collection_df einfügen
    result_df = collection_df.merge(
        client_rate[["client_id", "f_t_region_fraud_rate"]],
        on="client_id",
        how="left"
    )
    return result_df


def add_median_billing_frequence_per_region(
    merge_df: pd.DataFrame, collection_df: pd.DataFrame
) -> pd.DataFrame:
    # Copy to avoid modifying original
    df_copy = merge_df.copy()
    # Convert invoice_date to datetime
    df_copy["invoice_date"] = pd.to_datetime(
        df_copy["invoice_date"], format="%Y-%m-%d", errors="coerce"
    )
    # Sort by region, client_id, and date for correct diff calculation
    df_copy = df_copy.sort_values(["region", "client_id", "invoice_date"])
    # Compute days between consecutive invoices per client within each region
    df_copy["delta_days"] = (
        df_copy
        .groupby(["region", "client_id"])["invoice_date"]
        .diff()
        .dt.days
    )
    # Aggregate median interval per region
    median_region = (
        df_copy
        .groupby("region")["delta_days"]
        .median()
        .rename("f_region_median_billing_frequence_per")
        .reset_index()
    )
    # Map each client to its region's median frequency
    client_region = merge_df[["client_id", "region"]].drop_duplicates()
    client_median = client_region.merge(
        median_region, on="region", how="left"
    )
    # Merge the feature into the collection DataFrame on client_id
    result_df = collection_df.merge(
        client_median[["client_id", "f_region_median_billing_frequence_per"]],
        on="client_id",
        how="left"
    )
    return result_df


# ---------------------------------------------------------------------- #
# New feature: f_region_std_deviation_consumption
# ---------------------------------------------------------------------- #
def add_sdt_dev_consumption_region(
    merge_df: pd.DataFrame,
    collection_df: pd.DataFrame,
    postfix_consumption: str
) -> pd.DataFrame:
    # Copy to avoid modifying original
    df_copy = merge_df.copy()

    # Dynamische Spalten- und Feature-Namen bauen
    base_col = "consommation"
    col_name = f"{base_col}{postfix_consumption}"
    feature_name = f"f_region_std_deviation_consumption{postfix_consumption}"

    # Absicherung: Spalte muss existieren
    if col_name not in df_copy.columns:
        raise KeyError(f"Column '{col_name}' not found in merge_df")

    # 1) Std-Abweichung pro Region berechnen
    std_region = (
        df_copy
        .groupby("region")[col_name]
        .std()
        .rename(feature_name)
        .reset_index()
    )

    # 2) Jedem Client seine Region zuordnen (einmalig)
    client_region = df_copy[["client_id", "region"]].drop_duplicates()
    client_std = client_region.merge(
        std_region, on="region", how="left"
    )

    # 3) Feature ins collection_df mergen
    result_df = collection_df.merge(
        client_std[["client_id", feature_name]],
        on="client_id",
        how="left"
    )
    return result_df

def add_consump_agg(feature_dataframe, invoice) :

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

def add_tarif_agg(feature_dataframe, invoice):

    df_copy = invoice[['client_id', 'tarif_type']].copy()
    
    aggregated = df_copy.groupby('client_id')['tarif_type'].agg(
        lambda x: pd.Series.mode(x)[0]
    ).reset_index()
     
    feature_dataframe = feature_dataframe.merge(aggregated, on='client_id', how='left')
    
    return  feature_dataframe

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

def visualize_mutual_information(mi_results, title='MI on each feature', 
                                 xlabel='Mutual Information', ylabel='Feature',
                                 figsize=(16, 10), top_n=None, show=True):
    # Set a Top_n 
    if top_n is not None and top_n < len(mi_results):
        plot_data = mi_results.head(top_n).copy()
    else:
        plot_data = mi_results.copy()
    
    # Plot a graph 
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='MI_Score', y='Feature', data=plot_data)
    
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    
    # Set axis tick font sizes
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    # Show a value index
    for i, v in enumerate(plot_data['MI_Score']):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=18)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return plt.gcf()

def add_district_target_agg(feature_dataframe, client) :

    df_copy = client.copy()

    analysis_cols = ['target']
    
    # aggregate by district and calculate the mean of target variable
    aggregated_df = df_copy.groupby('disrict')[analysis_cols].mean().reset_index()

    # rename column
    aggregated_df.rename(columns={'target': 'f_t_district_target_mean'}, inplace=True)

    # merge to the feature_dataframe
    feature_dataframe = feature_dataframe.merge(aggregated_df, on='disrict', how='left')

    return feature_dataframe

def add_client_catg_target_agg(feature_dataframe, client) :

    df_copy = client.copy()

    analysis_cols = ['target']
    
    # aggregate by client_catg and calculate the mean of target variable
    aggregated_df = df_copy.groupby('client_catg')[analysis_cols].mean().reset_index()

    # rename column
    aggregated_df.rename(columns={'target': 'f_t_client_catg_target_mean'}, inplace=True)

    # merge to the feature_dataframe
    feature_dataframe = feature_dataframe.merge(aggregated_df, on='client_catg', how='left')

    return feature_dataframe

def add_index_cons_error_agg(feature_dataframe, invoice) :

    df_copy = invoice.copy()

    df_copy['index_cons_error'] = ((df_copy['new_index'] - df_copy['old_index']) -
                                  (df_copy['consommation_level_1'] + df_copy['consommation_level_2'] +
                                   df_copy['consommation_level_3'] + df_copy['consommation_level_4']))

    analysis_cols = ['index_cons_error']
    
    # aggregate by client_id and calculate the sum
    aggregated_df = df_copy.groupby('client_id')[analysis_cols].sum().reset_index()

    # rename column
    aggregated_df.rename(columns={'index_cons_error': 'f_index_cons_error_sum'}, inplace=True)

    # merge to the feature_dataframe
    feature_dataframe = feature_dataframe.merge(aggregated_df, on='client_id', how='left')

    return feature_dataframe


def add_counter_statue_agg(feature_dataframe, invoice) :

    df_copy = invoice.copy()

    # replace string values in counter_statue with numerical values of 0 to 5
    df_copy['counter_statue'] = df_copy['counter_statue'].replace({
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
    })

    # replace 'counter_statue' values that out of range (0,6) with None:
    df_copy['counter_statue'] = df_copy['counter_statue'].apply(lambda x: x if x in [0, 1, 2, 3, 4, 5] else None)

    # fill None values in 'counter_statue' with the mode of the column:
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='most_frequent')
    df_copy['counter_statue'] = imputer.fit_transform(df_copy[['counter_statue']])

    analysis_cols = ['counter_statue']
    
    # aggregate by client_id and calculate the mean
    aggregated_df = df_copy.groupby('client_id')[analysis_cols].mean().reset_index()

    # rename column
    aggregated_df.rename(columns={'counter_statue': 'f_counter_statue_mean'}, inplace=True)

    # merge to the feature_dataframe
    feature_dataframe = feature_dataframe.merge(aggregated_df, on='client_id', how='left')

    return feature_dataframe



# ---------------------------------------------------------------------- #
# Utility: merge_on
# ---------------------------------------------------------------------- #
def merge_on(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: str,
    right_column: str
) -> pd.DataFrame:
    
    if on not in left_df.columns:
        raise KeyError(f"'{on}' not found in left_df columns")
    if on not in right_df.columns:
        raise KeyError(f"'{on}' not found in right_df columns")
    if right_column not in right_df.columns:
        raise KeyError(f"'{right_column}' not found in right_df columns")

    to_merge = right_df[[on, right_column]].drop_duplicates()

    merged = left_df.merge(to_merge, on=on, how="left")
    return merged


def add_client_tenure(
    merge_df: pd.DataFrame,
    feature_df: pd.DataFrame
) -> pd.DataFrame:
    # Ensure datetime types
    merge_df['creation_date'] = pd.to_datetime(merge_df['creation_date'])
    merge_df['invoice_date'] = pd.to_datetime(merge_df['invoice_date'])

    # Aggregate to compute max invoice_date and first creation_date per client
    agg = merge_df.groupby('client_id').agg(
        max_invoice_date=('invoice_date', 'max'),
        first_creation_date=('creation_date', 'first')
    ).reset_index()

    # Calculate tenure in days
    agg['f_client_tenure_days'] = (agg['max_invoice_date'] - agg['first_creation_date']).dt.days

    # Merge back into feature_df
    result = feature_df.merge(
        agg[['client_id', 'f_client_tenure_days']],
        on='client_id', how='left'
    )
    return result

def add_meter_replacement_count_agg(feature_dataframe, invoice):
    df_copy = invoice.copy()
    
    # aggregate by client_id and calculate the unique count of counter_number
    aggregated_df = df_copy.groupby('client_id')['counter_number'].nunique().reset_index()

    # rename column
    aggregated_df.rename(columns={'counter_number': 'f_counter_number_nunique'}, inplace=True)

    # merge to the feature_dataframe
    feature_dataframe = feature_dataframe.merge(aggregated_df, on='client_id', how='left')

    return feature_dataframe

def add_tarif_change_count_agg(feature_dataframe, invoice):
    df_copy = invoice.copy()
    
    # aggregate by client_id and calculate the unique count of tarif_type
    aggregated_df = df_copy.groupby('client_id')['tarif_type'].nunique().reset_index()

    # rename column
    aggregated_df.rename(columns={'tarif_type': 'f_tarif_change_count'}, inplace=True)

    # merge to the feature_dataframe
    feature_dataframe = feature_dataframe.merge(aggregated_df, on='client_id', how='left')

    return feature_dataframe

import pandas as pd

def add_avg_consumption_per_month(merged, feature_dataframe):
    df = merged.copy()

    # 1. Gesamtverbrauch pro Rechnung berechnen
    df['total_consumption'] = (
        df['consommation_level_1'] +
        df['consommation_level_2'] +
        df['consommation_level_3'] +
        df['consommation_level_4']
    )

    # 2. Aggregation je Client
    agg_df = df.groupby('client_id').agg(
        total_consumption=('total_consumption', 'sum'),
        total_months=('months_number', 'sum')
    ).reset_index()

    # 3. Durchschnitt pro Monat berechnen
    agg_df['avg_consumption_per_month'] = agg_df['total_consumption'] / agg_df['total_months']

    # 4. Merge mit Feature-DataFrame
    result_df = feature_dataframe.merge(
        agg_df[['client_id', 'avg_consumption_per_month']],
        on='client_id',
        how='left'
    )

    return result_df



def add_reading_remarque_signals(merged, feature_dataframe):
    df = merged.copy()

    # Erzeuge Hilfsspalten
    df['has_remarque'] = df['reading_remarque'].notna() & (df['reading_remarque'].astype(str).str.strip() != '')
    df['remarque_length'] = df['reading_remarque'].astype(str).str.len().where(df['has_remarque'])

    # Aggregation pro Client
    agg_df = df.groupby('client_id').agg(
        total_invoices=('reading_remarque', 'count'),
        num_remarques=('has_remarque', 'sum'),
        avg_remarque_length=('remarque_length', 'mean')
    ).reset_index()

    # Verhältnis berechnen
    agg_df['remarque_frequency'] = agg_df['num_remarques'] / agg_df['total_invoices']

    # Auswahl + Merge
    result_df = feature_dataframe.merge(
        agg_df[['client_id', 'remarque_frequency', 'avg_remarque_length']],
        on='client_id',
        how='left'
    )

    return result_df

def collectAllFeaturesBaseline():
    fraud = Fraud(["./data/train/client_train.csv", "./data/train/invoice_train.csv"], target_column="target")
    client  = fraud["./data/train/client_train.csv"]
    invoice = fraud["./data/train/invoice_train.csv"]
    fraud_merged = left_join_on("client_id", client, invoice)
    feature_dataframe = fraud.get_target(client)
    print(feature_dataframe.head())
    feature_dataframe = add_invoice_frequency_features(fraud_merged, feature_dataframe)
    feature_dataframe = add_counter_statue_error_occured_features(fraud_merged, feature_dataframe)
    feature_dataframe = add_counter_regions_features(fraud_merged, feature_dataframe)
    feature_dataframe = add_median_billing_frequence_per_region(fraud_merged, feature_dataframe)
    feature_dataframe = add_sdt_dev_consumption_region(fraud_merged, feature_dataframe, postfix_consumption="_level_1")
    feature_dataframe = add_sdt_dev_consumption_region(fraud_merged, feature_dataframe, postfix_consumption="_level_2")
    feature_dataframe = add_sdt_dev_consumption_region(fraud_merged, feature_dataframe, postfix_consumption="_level_3")
    feature_dataframe = add_sdt_dev_consumption_region(fraud_merged, feature_dataframe, postfix_consumption="_level_4")
    feature_dataframe = add_consump_agg(feature_dataframe, invoice)
    feature_dataframe = add_tarif_agg(feature_dataframe, invoice)
    feature_dataframe = add_index_cons_error_agg(feature_dataframe, invoice)
    feature_dataframe = add_counter_statue_agg(feature_dataframe, invoice)
    feature_dataframe = add_client_tenure(fraud_merged, feature_dataframe)
    feature_dataframe = add_meter_replacement_count_agg(feature_dataframe, invoice)
    feature_dataframe = add_tarif_change_count_agg(feature_dataframe, invoice)
    feature_dataframe = add_avg_consumption_per_month(fraud_merged, feature_dataframe)
    feature_dataframe = add_reading_remarque_signals(fraud_merged, feature_dataframe)
    feature_dataframe = add_faulty_status_rate(fraud_merged, feature_dataframe)
    # with target together aggregated
    feature_dataframe = add_region_fraud_rate_features(fraud_merged, feature_dataframe)
    feature_dataframe = add_district_target_agg(feature_dataframe, client)
    feature_dataframe = add_client_catg_target_agg(feature_dataframe, client)


    return feature_dataframe

def add_faulty_status_rate(merged, feature_dataframe):
    df = merged.copy()

    # 'faulty'-Status erkennen (Groß-/Kleinschreibung ignorieren, NaN sicher behandeln)
    df['is_faulty'] = df['counter_statue'].astype(str).str.lower().eq('faulty')

    # Aggregation
    agg_df = df.groupby('client_id').agg(
        total_invoices=('counter_statue', 'count'),
        faulty_count=('is_faulty', 'sum')
    ).reset_index()

    # Verhältnis berechnen
    agg_df['faulty_status_rate'] = agg_df['faulty_count'] / agg_df['total_invoices']

    # Merge mit feature_dataframe
    result_df = feature_dataframe.merge(
        agg_df[['client_id', 'faulty_status_rate']],
        on='client_id',
        how='left'
    )

    return result_df

def collectAllFeaturesBaselineTest():
    fraud = Fraud(["./data/test/client_test.csv", "./data/test/invoice_test.csv"], None)
    client  = fraud["./data/test/client_test.csv"]
    invoice = fraud["./data/test/invoice_test.csv"]
    fraud_merged = left_join_on("client_id", client, invoice)
    feature_dataframe = fraud.get_target(client)
    print(feature_dataframe.head())
    feature_dataframe = add_invoice_frequency_features(fraud_merged, feature_dataframe)
    feature_dataframe = add_counter_statue_error_occured_features(fraud_merged, feature_dataframe)
    feature_dataframe = add_counter_regions_features(fraud_merged, feature_dataframe)
    feature_dataframe = add_median_billing_frequence_per_region(fraud_merged, feature_dataframe)
    feature_dataframe = add_sdt_dev_consumption_region(fraud_merged, feature_dataframe, postfix_consumption="_level_1")
    feature_dataframe = add_sdt_dev_consumption_region(fraud_merged, feature_dataframe, postfix_consumption="_level_2")
    feature_dataframe = add_sdt_dev_consumption_region(fraud_merged, feature_dataframe, postfix_consumption="_level_3")
    feature_dataframe = add_sdt_dev_consumption_region(fraud_merged, feature_dataframe, postfix_consumption="_level_4")
    feature_dataframe = add_consump_agg(feature_dataframe, invoice)
    feature_dataframe = add_tarif_agg(feature_dataframe, invoice)
    feature_dataframe = add_index_cons_error_agg(feature_dataframe, invoice)
    feature_dataframe = add_counter_statue_agg(feature_dataframe, invoice)
    feature_dataframe = add_client_tenure(fraud_merged, feature_dataframe)
    feature_dataframe = add_meter_replacement_count_agg(feature_dataframe, invoice)
    feature_dataframe = add_tarif_change_count_agg(feature_dataframe, invoice)
    feature_dataframe = add_avg_consumption_per_month(fraud_merged, feature_dataframe)
    feature_dataframe = add_reading_remarque_signals(fraud_merged, feature_dataframe)
    feature_dataframe = add_faulty_status_rate(fraud_merged, feature_dataframe)

    return feature_dataframe

def addCombinations(feature_dataframe):
    feature_dataframe = combine_features(feature_dataframe, "avg_remarque_length", "f_counter_number_nunique", "*","f_combi1")
    feature_dataframe = combine_features(feature_dataframe, "remarque_frequency", "f_tarif_change_count", "*","f_combi2")
    feature_dataframe = combine_features(feature_dataframe, "f_total_consumption_std", "f_counter_statue_error_occured", "/","f_combi3")
    feature_dataframe = combine_features(feature_dataframe,"f_region_std_deviation_consumption_level_3", "f_index_diff_mean", "*", "f_combi4")

    feature_dataframe = combine_features(feature_dataframe, "f_counter_number_nunique", "remarque_frequency", "*","f_combi5")
    feature_dataframe = combine_features(feature_dataframe, "f_counter_statue_error_occured", "remarque_frequency", "*","f_combi6")
    feature_dataframe = combine_features(feature_dataframe,"f_index_diff_mean", "remarque_frequency", "*", "f_combi7")

    return feature_dataframe


def filter_feature_names_by_mi(df, min_mi, max_mi):
    df_copy = df.copy()
    
    # Falls Spaltennamen fehlen oder falsch sind
    if list(df_copy.columns) != ['feature', 'mi']:
        df_copy.columns = ['feature', 'mi']
    
    # Filtern und Liste extrahieren
    filtered = df_copy[(df_copy['mi'] >= min_mi) & (df_copy['mi'] <= max_mi)]
    return filtered['feature'].tolist()

def combine_features(df, feature1, feature2, operation, name):
    if feature1 not in df.columns or feature2 not in df.columns:
        raise ValueError(f"One or both features '{feature1}', '{feature2}' not in DataFrame.")

    if operation == '+':
        df[name] = df[feature1] + df[feature2]
    elif operation == '-':
        df[name] = df[feature1] - df[feature2]
    elif operation == '*':
        df[name] = df[feature1] * df[feature2]
    elif operation == '/':
        df[name] = df[feature1] / (df[feature2] + 0.1)  # avoid division by zero by adding small constant
    else:
        raise ValueError("Operation must be one of '+', '-', '*', '/'")

    # Move the new column to the end to preserve order control
    columns = list(df.columns)
    columns.append(columns.pop(columns.index(name)))
    df = df[columns]

    return df