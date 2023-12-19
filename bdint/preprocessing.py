def filter_gr_livarea(df):
    # the motivation is in the Project Documentation. There is suggested to filter outliers over 4000
    size_A = len(df)
    df = df[df["GrLivArea"] < 4000]
    size_B = len(df)
    return df, (size_A - size_B)
