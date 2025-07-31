"""
w1_feature_fraud_mk.py

Utility objects for the fraud‑detection project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Union

import pandas as pd


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
                raise KeyError(f"Target column '{target_column}' not found in any loaded DataFrame")

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

    def get_target(self) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
         - client_id
         - <your target column name>
        """
        if self._target is None:
            raise ValueError("No target column was set at initialization")
        # self._target ist eine pd.Series mit Index=client_id
        # Mit reset_index() wandelt man beides in Spalten um
        df_target = self._target.rename("target_column_name").reset_index()
        # Optional: Spalten umbenennen, falls du statt "level_0" und dem Kolumnennamen
        # eigene Bezeichnungen möchtest:
        df_target.columns = ["client_id", self._target.name]
        return df_target


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
        .rename("f_region_fraud_rate")
        .reset_index()
    )

    # 2) Jedem Client seine Region-Rate zuordnen
    client_region = merge_df[["client_id", "region"]].drop_duplicates()
    client_rate = client_region.merge(
        region_rates, on="region", how="left"
    )

    # 3) In collection_df einfügen
    result_df = collection_df.merge(
        client_rate[["client_id", "f_region_fraud_rate"]],
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