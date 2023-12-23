import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from bdint.models.basemodel import BaseModel


def get_train_df(train_path="data/train.csv"):
    df = pd.read_csv(train_path)

    return df


def get_test_df(test_path="data/test.csv"):
    test_df = pd.read_csv(test_path)

    return test_df


def make_kaggle_submission_file(prediction, test_df):
    assert len(prediction) == test_df.shape[0]
    submission_df = test_df.copy().drop(columns=test_df.columns[1:])
    submission_df["SalePrice"] = prediction
    submission_df.to_csv("current_submission.csv", index=False)


def calculate_rmse(test, prediciton):
    s = []
    for idx, t in enumerate(test):
        a = t
        b = prediciton.iloc[idx]
        sqr = (a - b) ** 2
        s.append(sqr)
    return np.sqrt(np.sum(s) / len(s))


def k_fold_validation(train_df, model: BaseModel, k=10):
    target = train_df[["SalePrice"]]
    features = train_df.drop(columns=["SalePrice"])
    target.columns = ["SalePrice"]

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse_values = []

    for train_index, test_index in kf.split(features):
        x_train, x_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        # Learn the given model
        model.learn(x_train, y_train)

        # Calculate RMSE
        predictions = model.predict(x_test)
        rmse_value = mean_squared_error(y_test["SalePrice"], predictions, squared=False)
        rmse_values.append(rmse_value)

    # Calculate the mean RMSE
    mean_rmse = sum(rmse_values) / len(rmse_values)

    return mean_rmse


def preprocessing(df):
    # Remove uncorralated numerical features

    df = df.drop(["Id", "BsmtFinSF2"], axis=1)

    return df
