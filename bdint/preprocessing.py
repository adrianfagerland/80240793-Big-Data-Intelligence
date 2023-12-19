def filter_grlivarea(df):
    # the motivation is in the Project Documentation. There is suggested to filter outliers over 4000
    size_A = len(df)
    df = df[df["GrLivArea"] < 4000]
    size_B = len(df)
    return df, (size_A - size_B)

def cast_numerical_to_categorical(df, to_cast: list): #[]'MSSubClass', 'MoSold', 'YrSold']
    df.loc[:,to_cast] = df.loc[:,to_cast].astype('object')
    return df

def impute_na_by_none(df, features: list): #['Alley', 'GarageType', 'MiscFeature'] (nominal features)
    impute = df.loc[:,features]
    for i in impute.columns:
        df[i].fillna('None', inplace=True)
    return df

