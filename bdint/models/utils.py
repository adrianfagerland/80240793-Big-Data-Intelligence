import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class OHE:
    enc: OneHotEncoder

    def ohe(
        self, df, categorical_columns_parameter=None
    ) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
        df = df.copy()

        if categorical_columns_parameter is None:
            categorical_columns = df.select_dtypes(
                include=["category"]
            ).columns.tolist()

            self.enc = OneHotEncoder(handle_unknown="ignore")
            self.enc.fit(df[categorical_columns])
        else:
            categorical_columns = categorical_columns_parameter

        df_ohe = self.enc.transform(df[categorical_columns])

        # Convert sparse matrix to Pandas DataFrame
        df_ohe = pd.DataFrame(
            df_ohe.toarray(), columns=self.enc.get_feature_names_out(categorical_columns)  # type: ignore
        )

        # Combine with original numerical columns
        df_numerical = (df.select_dtypes(exclude=["category"]).reset_index(),)

        df_final = pd.concat([df_numerical, df_ohe], axis=1).fillna(0)

        if categorical_columns_parameter is None:
            return df_final, categorical_columns
        return df_final


def preprocess_for_numerical_model(df: pd.DataFrame) -> pd.DataFrame:
    preprocessed_df = df.copy().fillna(0).select_dtypes(include=["float64", "int64"])
    remove_columns = ["PoolQC", "Id"]
    remove_columns = [col for col in remove_columns if col in preprocessed_df.columns]
    return preprocessed_df.drop(columns=remove_columns)
