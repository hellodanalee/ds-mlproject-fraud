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


class Fraud:
    """
    Container that lazily reads a list of CSV files and stores them in
    a hash‑map‑like structure (``dict``) mapping *file path* → *pd.DataFrame*.

    Examples
    --------
    >>> f = Fraud(["data/client.csv", "data/invoice.csv"])
    >>> isinstance(f["data/client.csv"], pd.DataFrame)
    True
    >>> len(f)
    2
    """

    def __init__(
        self,
        csv_files: Union[Iterable[str], Iterable[Path]],
        target_column: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        csv_files
            Iterable with paths (``str`` or ``pathlib.Path``) to CSV files.

        Notes
        -----
        * Files are read eagerly via :pyfunc:`pandas.read_csv`.
        * Duplicate paths (after resolving) are ignored.
        """
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
    """
    Perform a **left join** between two DataFrames on a given column name.

    Parameters
    ----------
    key
        Column on which to join.
    left_df, right_df
        The two DataFrames to merge. ``left_df`` is considered the *primary*
        table (all seine Zeilen bleiben erhalten).
    suffixes
        Suffixes to apply to overlapping column names other than *key*.
    validate
        Passed through to :pyfunc:`pandas.merge` to enforce merge expectations.
        Use ``None`` to skip validation.

    Returns
    -------
    pandas.DataFrame
        Resulting DataFrame with columns from *left* plus matching columns
        from *right*.

    Examples
    --------
    >>> joined = left_join_on("client_id", client_df, invoice_df)
    >>> joined.shape
    (1000, 25)
    """
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
    """
    Add invoice frequency features to the DataFrame.

    Parameters
    ----------
    merge_df : pd.DataFrame
        DataFrame containing 'client_id' and 'invoice_date' columns to compute frequency on.
    collection_df : pd.DataFrame
        DataFrame (e.g., merged client table) to which the aggregated features will be added.

    Returns
    -------
    pd.DataFrame
        The `collection_df` extended with:
        - f_invoive_date_diff_days
        - f_invoive_date_median_months
        - f_invoive_date_median_years
    """
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
    """
    Add a binary feature indicating whether any non-zero counter_status occurred.

    Parameters
    ----------
    merge_df : pd.DataFrame
        DataFrame containing 'client_id' and 'counter_statue' columns.
    collection_df : pd.DataFrame
        DataFrame (e.g., merged client table) to which the aggregated feature will be added.

    Returns
    -------
    pd.DataFrame
        The `collection_df` extended with:
        - f_counter_statue_error_occured: 1 if any counter_statue != '0', else 0.
    """
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
    """
    Add a binary feature indicating whether a client spans multiple regions.

    Parameters
    ----------
    merge_df : pd.DataFrame
        DataFrame containing 'client_id' and 'region' columns.
    collection_df : pd.DataFrame
        DataFrame (e.g., merged client table) to which the aggregated feature will be added.

    Returns
    -------
    pd.DataFrame
        The `collection_df` extended with:
        - f_counter_regions: 1 if the client has more than one unique region, else 0.
    """
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
    """
    Add the fraud rate per region as a feature.
    ...
    """
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


# ---------------------------------------------------------------------- #
# New feature: f_median_billing_frequence_per_region
# ---------------------------------------------------------------------- #
def add_median_billing_frequence_per_region(
    merge_df: pd.DataFrame, collection_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add median billing frequency per region as a feature.

    Parameters
    ----------
    merge_df : pd.DataFrame
        DataFrame containing 'client_id', 'invoice_date', and 'region' columns.
    collection_df : pd.DataFrame
        DataFrame (e.g., merged client table) to which the aggregated feature will be added.

    Returns
    -------
    pd.DataFrame
        The `collection_df` extended with:
        - f_median_billing_frequence_per_region: median interval in days
          between consecutive invoices of all customers in each region.
    """
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
    """
    Add standard deviation of consumption values per region as a feature.

    Parameters
    ----------
    merge_df : pd.DataFrame
        DataFrame containing 'client_id', 'region' und dynamische Verbrauchs-Spalte.
    collection_df : pd.DataFrame
        DataFrame (z.B. Kunden-Tabelle), dem das Feature hinzugefügt wird.
    postfix_consumption : str
        Suffix, das an den Basis-Spaltennamen 'consumption' und an den Feature-Namen angehängt wird.

    Returns
    -------
    pd.DataFrame
        Das `collection_df` mit folgender neuer Spalte:
        - f_region_std_deviation_consumption<postfix>: Standardabweichung der Verbrauchswerte pro Region.
    """
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
    """
    Compute client tenure as days between creation_date and most recent invoice_date.

    Parameters
    ----------
    merge_df : pd.DataFrame
        Left-merged client-to-invoice DataFrame containing 'client_id', 'creation_date', and 'invoice_date'.
    feature_df : pd.DataFrame
        DataFrame of client-level features to which 'client_tenure_days' will be added.

    Returns
    -------
    pd.DataFrame
        feature_df with a new column 'client_tenure_days'.
    """
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


def collectAllFeaturesBaseline():
    """
    Collect all features into a single DataFrame.

    Parameters
    ----------
    feature_dataframe : pd.DataFrame
        The DataFrame to which features will be added.
    client : pd.DataFrame
        Client DataFrame.
    invoice : pd.DataFrame
        Invoice DataFrame.
    fraud_merged : pd.DataFrame
        Merged DataFrame containing fraud information.

    Returns
    -------
    pd.DataFrame
        The `feature_dataframe` extended with all collected features.
    """
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
    # with target together aggregated
    feature_dataframe = add_region_fraud_rate_features(fraud_merged, feature_dataframe)
    feature_dataframe = add_district_target_agg(feature_dataframe, client)
    feature_dataframe = add_client_catg_target_agg(feature_dataframe, client)


    return feature_dataframe


def collectAllFeaturesBaselineTest():
    """
    Collect all features into a single DataFrame.

    Parameters
    ----------
    feature_dataframe : pd.DataFrame
        The DataFrame to which features will be added.
    client : pd.DataFrame
        Client DataFrame.
    invoice : pd.DataFrame
        Invoice DataFrame.
    fraud_merged : pd.DataFrame
        Merged DataFrame containing fraud information.

    Returns
    -------
    pd.DataFrame
        The `feature_dataframe` extended with all collected features.
    """
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
    

    return feature_dataframe