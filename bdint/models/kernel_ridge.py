import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge as KernelRidgeRegressor

from bdint.models.utils import OHE
from bdint.preprocessing import (
    cast_numerical_to_categorical,
    categorical_data_to_label_encoding,
    impute_na_by_mode,
    impute_na_by_none,
    impute_numerical_by_median,
    impute_numerical_by_zero,
    log_transform_if_skewed,
)

from .basemodel import BaseModel


class KernelRidgeRegression(BaseModel):
    categorical_columns = None

    def __init__(
        self,
        train,
        test,
        skewness_threshold=0.7,
        **kwargs,
    ):
        self.model = KernelRidgeRegressor(**kwargs)
        self.kwargs = kwargs
        self.skewness_threshold = skewness_threshold
        self.ohe = OHE()
        self.train, self.test = self._preprocess(train_df=train, test_df=test)
        self.skewness_threshold = skewness_threshold

    def _preprocess(self, train_df, test_df):
        all_df = pd.concat([train_df, test_df], axis=0, sort=True)

        # drop uncoraleted Columns
        to_drop = ["Id"]  # Id bc Id
        # to_drop = to_drop + ["PoolQC"]

        all_df = all_df.drop(to_drop, axis=1)

        # cast numericals which should be categorical
        all_df = cast_numerical_to_categorical(df=all_df, to_cast=["MSSubClass", "MoSold", "YrSold"])

        # Missing Values: Numericals Which Can be imputed by Mean
        all_df = impute_numerical_by_median(
            all_df,
            [
                "BsmtFullBath",
                "BsmtHalfBath",
                "GarageCars",
                "MasVnrArea",
                "GarageYrBlt",
                "BsmtFinSF1",
                "BsmtFinSF2",
                "BsmtUnfSF",
                "TotalBsmtSF",
                "GarageArea",
                "LotFrontage",
            ],
        )

        # Missing Values: Numericals Which Can be imputed by Zero
        all_df = impute_numerical_by_zero(all_df, ["MasVnrArea"])

        # Missing Values: Categorical which can be imputed by Mode (most frequent category)
        all_df = impute_na_by_mode(
            all_df,
            [
                "Electrical",
                "MSZoning",
                "Utilities",
                "Exterior1st",
                "Exterior2nd",
                "KitchenQual",
                "Functional",
                "SaleType",
            ],
        )

        # Missing Values: Categorical which can be imputed by NONE
        all_df = impute_na_by_none(
            all_df,
            [
                "PoolQC",
                "MiscFeature",
                "Alley",
                "Fence",
                "FireplaceQu",
                "GarageType",
                "GarageCond",
                "GarageFinish",
                "GarageQual",
                "BsmtFinType2",
                "BsmtExposure",
                "BsmtQual",
                "BsmtCond",
                "BsmtFinType1",
                "MasVnrType",
            ],
        )

        # Categorical To Label Encoding
        all_df = categorical_data_to_label_encoding(all_df)

        # Feature Engeneering
        all_df["TotalSF"] = all_df["TotalBsmtSF"] + all_df["1stFlrSF"] + all_df["2ndFlrSF"]

        """all_df = all_df[
            [
                "OverallQual",
                "TotalSF",
                "GrLivArea",
                "ExterQual",
                "KitchenQual",
                "GarageCars",
                "BsmtQual",
                "Neighborhood",
                "Condition1",
                "Condition2",
                "SaleType",
                "MSSubClass",
                "Exterior2nd",
                "SalePrice",
            ]
        ]"""
        # print("MISSING COLUMNS:", all_df.columns[all_df.isna().any()].values)

        # transform skewness of numericals
        all_df = log_transform_if_skewed(all_df, self.skewness_threshold)

        # categorical_boxplot(pd.concat([df, train_df], axis=1))
        # numerical_scatter_regression(pd.concat([df, train_df], axis=1))

        # OHE
        categorical_columns = all_df.select_dtypes(include=["object"]).columns.tolist()
        all_df = pd.get_dummies(all_df, columns=categorical_columns, prefix=categorical_columns)

        train = all_df.iloc[0:1459, :]
        test = all_df.iloc[1460:, :]
        test = test.drop(["SalePrice"], axis=1)

        return train, test

    def learn(self, x_train_df=None, y_train_df=None):
        self.model = KernelRidgeRegressor(**self.kwargs)
        if x_train_df is None:
            x_train_df = self.train.drop(columns=["SalePrice"])
        if y_train_df is None:
            y_train_df = self.train["SalePrice"]

        self.model.fit(x_train_df, y_train_df)

    def predict(self, test_x=None):
        if test_x is None:
            test_x = self.test
        return np.expm1(self.model.predict(test_x))
