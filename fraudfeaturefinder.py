

from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import itertools

class fraudfeaturefinder:
    def __init__(
        self,
        df_feature: pd.DataFrame,
        target_col: str = "target",
        threshold: float = 0.025,
        ignore_cols=None,
        operations=None,
        selected_features=None
    ):
        self.df_feature = df_feature.copy()
        self.target_col = target_col
        self.threshold = threshold
        self.valid_operations = operations if operations is not None else ['+', '-', '*', '/']
        self.feature_pairs = []
        self.results = []
        self.ignore_cols = ignore_cols or []
        self.selected_features = selected_features

    def _generate_combinations(self):
        if self.selected_features:
            feature_columns = [
                col for col in self.selected_features
                if col in self.df_feature.columns and pd.api.types.is_numeric_dtype(self.df_feature[col])
            ]
        else:
            feature_columns = [
                col for col in self.df_feature.columns
                if col != self.target_col and col not in self.ignore_cols and pd.api.types.is_numeric_dtype(self.df_feature[col])
            ]
        self.feature_pairs = list(itertools.combinations(feature_columns, 2))
        print('numerber of feature pairs:', len(self.feature_pairs))

    def _apply_operation(self, col1, col2, operation):
        if operation == '+':
            return self.df_feature[col1] + self.df_feature[col2]
        elif operation == '-':
            return self.df_feature[col1] - self.df_feature[col2]
        elif operation == '*':
            return self.df_feature[col1] * self.df_feature[col2]
        elif operation == '/':
            return self.df_feature[col1] / (self.df_feature[col2] + 0.1)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def find_good_combinations(self):
        self._generate_combinations()
        y = self.df_feature[self.target_col]

        i =0
        for col1, col2 in self.feature_pairs:
            i = i+1
            for op in self.valid_operations:
                try:
                    new_feature = self._apply_operation(col1, col2, op)
                    if new_feature.isnull().all():
                        continue

                    X_temp = pd.DataFrame({f"{col1}{op}{col2}": new_feature}).fillna(0)
                    mi = mutual_info_classif(X_temp, y, random_state=0)[0]

                    if mi >= self.threshold:
                        self.results.append({
                            "feature_name": f"{col1}{op}{col2}",
                            "col1": col1,
                            "col2": col2,
                            "operation": op,
                            "mutual_info": mi
                        })
                        print(f"{i} {col1} {op} {col2} = {mi:.4f}")
                except Exception:
                    continue

    def save_results(self, output_path="./data/mutual-info-combinations.csv"):
        if self.results:
            df_results = pd.DataFrame(self.results)
            df_results = df_results.sort_values(by="mutual_info", ascending=False)
            df_results.to_csv(output_path, index=False)
        else:
            print("No combinations passed the threshold.")

            
def apply_saved_combinations(df_feature: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    combinations_df = pd.read_csv(csv_path)

    if "client_id" not in df_feature.columns:
        raise KeyError("'client_id' must be present in df_feature.")

    combined_features = pd.DataFrame()
    combined_features["client_id"] = df_feature["client_id"]

    counter = 1
    for _, row in combinations_df.iterrows():
        col1, col2, op = row["col1"], row["col2"], row["operation"]
        new_col_name = f"f_autocombi_{counter}"

        if col1 not in df_feature.columns or col2 not in df_feature.columns:
            raise KeyError(f"One or both columns '{col1}' and '{col2}' not found in dataframe.")

        if op == '+':
            combined_features[new_col_name] = df_feature[col1] + df_feature[col2]
        elif op == '-':
            combined_features[new_col_name] = df_feature[col1] - df_feature[col2]
        elif op == '*':
            combined_features[new_col_name] = df_feature[col1] * df_feature[col2]
        elif op == '/':
            combined_features[new_col_name] = df_feature[col1] / (df_feature[col2] + 0.1)
        else:
            raise ValueError(f"Unsupported operation: {op}")

        counter += 1

    df_feature = df_feature.merge(combined_features, on="client_id", how="left")
    return df_feature