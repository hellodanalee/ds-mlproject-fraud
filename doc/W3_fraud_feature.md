## Feature-Engineering Utility Functions (`add_*`)

Below is a list of all `add_…` functions in your `fraud.py`, along with a brief English description of what each does:

- **`add_invoice_frequency_features(merged_df, feature_df)`**  
  Computes per-client statistics on the time gaps between invoices (e.g. mean, median, min, max days between invoices) and merges them into `feature_df`.

- **`add_counter_status_error_occured_features(merged_df, feature_df)`**  
  Creates a binary flag indicating whether any meter status reading for a client was “faulty” or “error” at least once.

- **`add_counter_regions_features(merged_df, feature_df)`**  
  Flags clients whose invoices span more than one region (i.e. they have meter readings in multiple `region` values).

- **`add_region_fraud_rate_features(merged_df, feature_df)`**  
  Calculates the fraud rate (mean of the `target`) for each `region`, then joins that regional rate back into the client-level `feature_df`.

- **`add_median_billing_frequency_per_region(merged_df, feature_df)`**  
  Computes the median invoice‐interval (in days) for each region and adds that as a regional benchmark feature per client.

- **`add_std_dev_consumption_region(merged_df, feature_df)`**  
  Measures the standard deviation of consumption across all clients within each region and merges that regional variability back in.

- **`add_consump_agg(merged_df, feature_df)`**  
  Aggregates the four consumption tiers (`consommation_level_1`…`level_4`) per client into overall statistics (sum, mean, min, max, std).

- **`add_tarif_agg(merged_df, feature_df)`**  
  Identifies the most frequent `tarif_type` per client (the “mode”) and attaches it as a categorical feature.

- **`add_index_cons_error_agg(merged_df, feature_df)`**  
  Calculates, for each client, the total discrepancy between `(new_index - old_index)` and the reported consumption—anomalies suggest potential manipulation.

- **`add_counter_status_agg(merged_df, feature_df)`**  
  Computes per-client aggregates of the meter status codes (e.g., average or counts of each status) to capture reading‐quality patterns.

- **`add_district_target_agg(merged_df, feature_df)`**  
  Computes the fraud rate for each `district` (mean of `target`) and joins it as a district-level risk indicator per client.

- **`add_client_catg_target_agg(merged_df, feature_df)`**  
  Computes the fraud rate per `client_catg` (client category) and merges it as a category-level risk feature.