def filter_gr_livarea(df):
    # the motivation is in the Project Documentation. There is suggested to filter outliers over 4000
    df = df[df["GrLivArea"] < 4000]
    return df
