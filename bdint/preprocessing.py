from bdint.models.utils import OHE


def preprocessor(train, test):
    # NUMERICAL TO CATEGORICAL
    trian = cast_numerical_to_categorical(df=train, to_cast=["MSSubClass", "MoSold", "YrSold"])
    test = cast_numerical_to_categorical(df=test, to_cast=["MSSubClass", "MoSold", "YrSold"])

    # CATEGORICAL TO ORDINAL (Label encoding)

    # MISSING VALUES
    impute_na_by_none(train, ["Alley", "GarageType", "MiscFeature"])
    impute_na_by_none(test, ["Alley", "GarageType", "MiscFeature"])

    impute_numerical_by_median(train, ["Order", "Lot Frontage"])
    impute_numerical_by_median(test, ["Order", "Lot Frontage"])

    # CATEGORICAL TO OHE ()
    ohe = OHE()
    train_ohe, col = ohe.ohe(train)
    test_ohe = ohe.ohe(test, col)

    # OUTLIER
    filter_gr_livarea(train)


def cast_numerical_to_categorical(df, to_cast: list):  # []'MSSubClass', 'MoSold', 'YrSold']
    df.loc[:, to_cast] = df.loc[:, to_cast].astype("object")
    return df


def filter_gr_livarea(df):
    # the motivation is in the Project Documentation. There is suggested to filter outliers over 4000
    size_A = len(df)
    df = df[df["GrLivArea"] < 4000]
    size_B = len(df)
    return df, (size_A - size_B)


def impute_na_by_none(df, features: list):  # ['Alley', 'GarageType', 'MiscFeature'] (nominal features)
    impute = df.loc[:, features]
    for i in impute.columns:
        df[i].fillna("None", inplace=True)
    return df


def impute_numerical_by_median(df, features: list):  # ['Order', 'Lot Frontage']
    impute = df.loc[:, features]
    for i in impute.columns:
        df[i].fillna(df[i].median(), inplace=True)
    return df
