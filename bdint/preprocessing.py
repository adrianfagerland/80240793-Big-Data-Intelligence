import numpy as np
import pandas as pd

"""
def preprocessor(train, test):
    # NUMERICAL TO CATEGORICAL
    train = cast_numerical_to_categorical(df=train, to_cast=["MSSubClass", "MoSold", "YrSold"])
    test = cast_numerical_to_categorical(df=test, to_cast=["MSSubClass", "MoSold", "YrSold"])

    # CATEGORICAL TO ORDINAL (Label encoding)

    # MISSING VALUES
    train = impute_na_by_none(train, ["Alley", "GarageType", "MiscFeature"])
    test = impute_na_by_none(test, ["Alley", "GarageType", "MiscFeature"])

    train = impute_numerical_by_median(train, ["Order", "Lot Frontage"])
    test = impute_numerical_by_median(test, ["Order", "Lot Frontage"])

    # CATEGORICAL TO OHE ()
    ohe = OHE()
    train, col = ohe.ohe(train)
    test = ohe.ohe(test, col)

    # OUTLIER
    train = filter_gr_livarea(train)
    return train, test"""


def cast_numerical_to_categorical(df, to_cast: list):  # []"MSSubClass", "MoSold", "YrSold"]
    df.loc[:, to_cast] = df.loc[:, to_cast].astype("object")
    return df


def filter_gr_livarea(df):
    # the motivation is in the Project Documentation. There is suggested to filter outliers over 4000
    size_A = len(df)
    df = df[df["GrLivArea"] < 4000]
    size_B = len(df)
    return df, (size_A - size_B)


def impute_na_by_none(df, features: list):
    impute = df.loc[:, features]
    for i in impute.columns:
        df[i].fillna("None", inplace=True)
    return df


def impute_na_by_mode(df, features: list):
    impute = df.loc[:, features]
    for i in impute.columns:
        df[i].fillna(df[i].mode()[0], inplace=True)
    return df


def impute_numerical_by_median(df, features: list):  # ["Order", "Lot Frontage"]
    impute = df.loc[:, features]
    for i in impute.columns:
        df[i].fillna(df[i].median(), inplace=True)
    return df


def impute_numerical_by_zero(df, features: list):  # ["MasVnrArea"]
    impute = df.loc[:, features]
    for i in impute.columns:
        df[i].fillna(0, inplace=True)
    return df


def categorical_data_to_label_encoding(df):
    df.LotShape.replace(to_replace=["IR3", "IR2", "IR1", "Reg"], value=[0, 1, 2, 3], inplace=True)
    df.LandContour.replace(to_replace=["Low", "Bnk", "HLS", "Lvl"], value=[0, 1, 2, 3], inplace=True)
    df.Utilities.replace(to_replace=["NoSeWa", "AllPub"], value=[0, 1], inplace=True)
    df.LandSlope.replace(to_replace=["Sev", "Mod", "Gtl"], value=[0, 1, 2], inplace=True)
    df.ExterQual.replace(to_replace=["Fa", "TA", "Gd", "Ex"], value=[0, 1, 2, 3], inplace=True)
    df.ExterCond.replace(to_replace=["Po", "Fa", "TA", "Gd", "Ex"], value=[0, 1, 2, 3, 4], inplace=True)
    df.BsmtQual.replace(to_replace=["None", "Fa", "TA", "Gd", "Ex"], value=[0, 1, 2, 3, 4], inplace=True)
    df.BsmtCond.replace(to_replace=["None", "Po", "Fa", "TA", "Gd"], value=[0, 1, 2, 3, 4], inplace=True)
    df.BsmtExposure.replace(to_replace=["None", "No", "Mn", "Av", "Gd"], value=[0, 1, 2, 3, 4], inplace=True)
    df.BsmtFinType1.replace(
        to_replace=["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"], value=[0, 1, 2, 3, 4, 5, 6], inplace=True
    )
    df.BsmtFinType2.replace(
        to_replace=["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"], value=[0, 1, 2, 3, 4, 5, 6], inplace=True
    )
    df.HeatingQC.replace(to_replace=["Po", "Fa", "TA", "Gd", "Ex"], value=[0, 1, 2, 3, 4], inplace=True)
    df.Electrical.replace(to_replace=["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"], value=[0, 1, 2, 3, 4], inplace=True)
    df.KitchenQual.replace(to_replace=["Fa", "TA", "Gd", "Ex"], value=[0, 1, 2, 3], inplace=True)
    df.Functional.replace(
        to_replace=["Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"], value=[0, 1, 2, 3, 4, 5, 6], inplace=True
    )
    df.FireplaceQu.replace(to_replace=["None", "Po", "Fa", "TA", "Gd", "Ex"], value=[0, 1, 2, 3, 4, 5], inplace=True)
    df.GarageFinish.replace(to_replace=["None", "Unf", "RFn", "Fin"], value=[0, 1, 2, 3], inplace=True)
    df.GarageQual.replace(to_replace=["None", "Po", "Fa", "TA", "Gd", "Ex"], value=[0, 1, 2, 3, 4, 5], inplace=True)
    df.GarageCond.replace(to_replace=["None", "Po", "Fa", "TA", "Gd", "Ex"], value=[0, 1, 2, 3, 4, 5], inplace=True)
    df.PavedDrive.replace(to_replace=["N", "P", "Y"], value=[0, 1, 2], inplace=True)
    df.PoolQC.replace(to_replace=["None", "Fa", "Gd", "Ex"], value=[0, 1, 2, 3], inplace=True)
    df.Fence.replace(to_replace=["None", "MnWw", "GdWo", "MnPrv", "GdPrv"], value=[0, 1, 2, 3, 4], inplace=True)

    return df


def log_transform_if_skewed(df, skewness_threshold):
    df_num = df.select_dtypes(include=["int64", "float64"])
    df_non_num = df.select_dtypes(exclude=["int64", "float64"])

    # always log transform SalePrice
    df_num["SalePrice"] = np.log1p(df_num["SalePrice"])

    # remove SalePrice from df_num before calculating skewness
    df_num_without_saleprice = df_num.drop(columns=["SalePrice"])

    df_skewed = np.log1p(
        df_num_without_saleprice[
            df_num_without_saleprice.skew()[df_num_without_saleprice.skew() > skewness_threshold].index
        ]
    )

    df_non_skew = df_num_without_saleprice[
        df_num_without_saleprice.skew()[df_num_without_saleprice.skew() <= skewness_threshold].index
    ]

    # add SalePrice back to the dataframe
    df_final = pd.concat([df_skewed, df_non_skew, df_num["SalePrice"], df_non_num], axis=1)

    return df_final


"""def log_transform_if_skewed(series, skewness_threshold):
    if series.dtype in ["int64", "float64"]:
        skewness = series.skew()
        if abs(skewness) > skewness_threshold:
            print(round(abs(skewness), 2), " ", end="")
            return np.log1p(series)
        else:
            return series
    else:
        return series"""
