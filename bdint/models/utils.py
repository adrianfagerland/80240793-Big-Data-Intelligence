import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class OHE:
    enc: OneHotEncoder

    def ohe(
        self, df, categorical_columns_parameter=None, use_category=False
    ) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
        df = df.copy()

        if categorical_columns_parameter is None:
            categorical_columns = df.select_dtypes(include=["category" if use_category else "object"]).columns.tolist()

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
        df_numerical = df.select_dtypes(exclude=["category" if use_category else "object"]).reset_index()

        df_final = pd.concat([df_numerical, df_ohe], axis=1).fillna(0)

        if categorical_columns_parameter is None:
            return df_final, categorical_columns
        return df_final


def preprocess_for_numerical_model(df: pd.DataFrame) -> pd.DataFrame:
    preprocessed_df = df.copy().fillna(0).select_dtypes(include=["float64", "int64"])
    remove_columns = ["PoolQC", "Id"]
    remove_columns = [col for col in remove_columns if col in preprocessed_df.columns]
    return preprocessed_df.drop(columns=remove_columns)


def preprocess_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LotFrontage"] = df["LotFrontage"].fillna(-1).astype("int64")
    df["MasVnrArea"] = df["MasVnrArea"].fillna(-1).astype("int64")
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(-1).astype("int64")
    df.Electrical = df.Electrical.fillna("SBrkr")
    df.Utilities = df.Utilities.fillna("AllPub")
    df.BsmtFinSF1 = df.BsmtFinSF1.fillna(0)
    df.BsmtFinSF2 = df.BsmtFinSF2.fillna(0)
    df.BsmtUnfSF = df.BsmtUnfSF.fillna(0)
    df.TotalBsmtSF = df.TotalBsmtSF.fillna(0)
    df.BsmtFullBath = df.BsmtFullBath.fillna(0)
    df.BsmtHalfBath = df.BsmtHalfBath.fillna(0)
    df.KitchenQual = df.KitchenQual.fillna("TA")
    df.Functional = df.Functional.fillna("Typ")
    df.GarageCars = df.GarageCars.fillna(0)  # assuming that this is 0 if it is na
    df.GarageArea = df.GarageArea.fillna(0)  # assuming that this is 0 if it is na
    df = df.fillna("NA")
    df[["MSSubClass", "MoSold", "YrSold"]] = df[["MSSubClass", "MoSold", "YrSold"]].astype("category")
    df.LotShape = pd.Categorical(df.LotShape, categories=["IR3", "IR2", "IR1", "Reg"], ordered=True)
    df.LandContour = pd.Categorical(df.LandContour, categories=["Low", "Bnk", "HLS", "Lvl"], ordered=True)
    df.Utilities = pd.Categorical(df.Utilities, categories=["NoSeWa", "AllPub"], ordered=True)
    df.LandSlope = pd.Categorical(df.LandSlope, categories=["Sev", "Mod", "Gtl"], ordered=True)
    df.ExterQual = pd.Categorical(df.ExterQual, categories=["Fa", "TA", "Gd", "Ex"], ordered=True)
    df.ExterCond = pd.Categorical(df.ExterCond, categories=["Po", "Fa", "TA", "Gd", "Ex"], ordered=True)
    df.BsmtQual = pd.Categorical(df.BsmtQual, categories=["NA", "Fa", "TA", "Gd", "Ex"], ordered=True)
    df.BsmtCond = pd.Categorical(df.BsmtCond, categories=["NA", "Po", "Fa", "TA", "Gd"], ordered=True)
    df.BsmtExposure = pd.Categorical(df.BsmtExposure, categories=["NA", "No", "Mn", "Av", "Gd"], ordered=True)
    df.BsmtFinType1 = pd.Categorical(
        df.BsmtFinType1,
        categories=["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        ordered=True,
    )
    df.BsmtFinType2 = pd.Categorical(
        df.BsmtFinType2,
        categories=["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        ordered=True,
    )
    df.HeatingQC = pd.Categorical(df.HeatingQC, categories=["Po", "Fa", "TA", "Gd", "Ex"], ordered=True)
    df.Electrical = pd.Categorical(
        df.Electrical,
        categories=["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
        ordered=True,
    )
    df.KitchenQual = pd.Categorical(df.KitchenQual, categories=["Fa", "TA", "Gd", "Ex"], ordered=True)
    df.Functional = pd.Categorical(
        df.Functional,
        categories=["Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
        ordered=True,
    )
    df.FireplaceQu = pd.Categorical(df.FireplaceQu, categories=["NA", "Po", "Fa", "TA", "Gd", "Ex"], ordered=True)
    df.GarageFinish = pd.Categorical(df.GarageFinish, categories=["NA", "Unf", "RFn", "Fin"], ordered=True)
    df.GarageQual = pd.Categorical(df.GarageQual, categories=["NA", "Po", "Fa", "TA", "Gd", "Ex"], ordered=True)
    df.GarageCond = pd.Categorical(df.GarageCond, categories=["NA", "Po", "Fa", "TA", "Gd", "Ex"], ordered=True)
    df.PavedDrive = pd.Categorical(df.PavedDrive, categories=["N", "P", "Y"], ordered=True)
    df.PoolQC = pd.Categorical(df.PoolQC, categories=["NA", "Fa", "Gd", "Ex"], ordered=True)
    df.Fence = pd.Categorical(df.Fence, categories=["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"], ordered=True)
    df = df.apply(lambda col: col.astype("category") if col.dtype == "object" else col)
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    return df
