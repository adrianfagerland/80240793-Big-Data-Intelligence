import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def ohe(train_df: pd.DataFrame, test_df: pd.DataFrame):
    #so what this should do is first fit the OneHotEncoder on the trainset using handle_unknown = 'ignore'. THan we can transform the train and test set. The numerical features should not be transformed. Problem is, that the result is a nparray and not pd.DataFrame. I dont no how to get it back to a pandas df 
    
    train_df_copy : pd.DataFrame = train_df.copy()
    test_df_copy : pd.DataFrame = test_df.copy()

    categorical_columns = train_df_copy.select_dtypes(include=['category', 'object']).columns.tolist()


    enc = OneHotEncoder(handle_unknown = 'ignore', sparse=False)
    
    enc.fit(train_df_copy[categorical_columns])

    train_ohe = enc.transform(train_df_copy[categorical_columns])
    test_ohe = enc.transform(test_df_copy[categorical_columns])


    one_hot_df = pd.DataFrame(train_ohe, columns=ohe.categories_)


    return train_ohe, test_ohe



def preprocess_for_numerical_model(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    numerical_columns = df_copy.select_dtypes(include=['number']).columns
    df_copy[numerical_columns] = df_copy[numerical_columns].fillna(0)
    return df_copy


       
