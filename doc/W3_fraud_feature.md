# 📘 Feature Engineering Functions Overview

This document summarizes all feature-generating functions used in the `fraud.py` module of the fraud detection project.

---

### 1. `add_invoice_frequency_features()`

Computes time-based frequency features from `invoice_date`:
- `f_invoive_date_diff_days`: median days between invoices per client
- `f_invoive_date_median_months`: same in months
- `f_invoive_date_median_years`: same in years

---

### 2. `add_counter_statue_error_occured_features()`

Adds:
- `f_counter_statue_error_occured`: 1 if any non-zero `counter_statue` exists for the client, else 0

---

### 3. `add_counter_regions_features()`

Adds:
- `f_counter_regions`: 1 if the client has invoices linked to more than one unique region

---

### 4. `add_region_fraud_rate_features()`

Adds:
- `f_t_region_fraud_rate`: average fraud rate (target = 1) in the client's region

---

### 5. `add_median_billing_frequence_per_region()`

Adds:
- `f_region_median_billing_frequence_per`: median number of days between invoices in the client’s region

---

### 6. `add_sdt_dev_consumption_region(postfix)`

Adds:
- `f_region_std_deviation_consumption<postfix>`: standard deviation of consumption (e.g., `consommation_level_1`) per region

---

### 7. `add_consump_agg()`

Aggregates the following per client:
- `consommation_level_1-4`
- `f_index_diff`: difference between `new_index` and `old_index`
- `f_total_consumption`: sum of all levels  
For each, computes: min, max, std, mean

---

### 8. `add_tarif_agg()`

Adds:
- Most frequent `tarif_type` per client

---

### 9. `add_index_cons_error_agg()`

Adds:
- `f_index_cons_error_sum`: cumulative difference between physical index delta and total consumption

---

### 10. `add_counter_statue_agg()`

Adds:
- `f_counter_statue_mean`: mean of encoded `counter_statue` values (0–5)

---

### 11. `add_client_tenure()`

Adds:
- `f_client_tenure_days`: number of days between account creation and last invoice

---

### 12. `add_meter_replacement_count_agg()`

Adds:
- `f_counter_number_nunique`: number of unique meter (counter) replacements per client

---

### 13. `add_tarif_change_count_agg()`

Adds:
- `f_tarif_change_count`: number of unique `tarif_type` values per client

---

### 14. `add_avg_consumption_per_month()`

Adds:
- `avg_consumption_per_month`: total consumption across invoices divided by total months billed

---

### 15. `add_reading_remarque_signals()`

Adds:
- `remarque_frequency`: share of invoices with a non-empty `reading_remarque`
- `avg_remarque_length`: average number of characters in remarks (non-empty only)

---

### 16. `add_faulty_status_rate()`

Adds:
- `faulty_status_rate`: share of invoices where `counter_statue == "faulty"`

---