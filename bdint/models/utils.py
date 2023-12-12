import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def ohe(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()

    categorical_columns = train_df_copy.select_dtypes(
        include=["object"]
    ).columns.tolist()

    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(train_df_copy[categorical_columns])

    train_ohe = enc.transform(train_df_copy[categorical_columns])
    test_ohe = enc.transform(test_df_copy[categorical_columns])

    # Convert sparse matrix to Pandas DataFrame
    train_ohe_df = pd.DataFrame(
        train_ohe.toarray(), columns=enc.get_feature_names_out(categorical_columns)  # type: ignore
    )
    test_ohe_df = pd.DataFrame(
        test_ohe.toarray(), columns=enc.get_feature_names_out(categorical_columns)  # type: ignore
    )

    # Combine with original numerical columns
    train_numerical = train_df_copy.select_dtypes(exclude=["object"])
    test_numerical = test_df_copy.select_dtypes(exclude=["object"])

    train_final = pd.concat([train_numerical, train_ohe_df], axis=1).fillna(0)
    test_final = pd.concat([test_numerical, test_ohe_df], axis=1).fillna(0)

    return train_final, test_final


def preprocess_for_numerical_model(df: pd.DataFrame) -> pd.DataFrame:
    preprocessed_df = df.copy().fillna(0).select_dtypes(include=["float64", "int64"])
    remove_columns = ["PoolQC", "Id"]
    remove_columns = [col for col in remove_columns if col in preprocessed_df.columns]
    return preprocessed_df.drop(columns=remove_columns)
