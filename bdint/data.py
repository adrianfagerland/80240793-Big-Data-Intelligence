import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

class Data:
    def get_train_df(self, validation = None):
        train_path = "bdint/data/train.csv"
        df = pd.read_csv(train_path)
        
        train_df, validation_df = train_test_split(df, test_size=validation, random_state=187)
        return train_df, validation_df
    
    def get_test_df(self):
        test_path = "bdint/data/test.csv"
        test_df = pd.read_csv(test_path)
        
        return test_df
    
    def make_kaggle_submission_file(self, prediction_df):
        path = "kaggle/submission.csv"
        assert len(prediction_df) == 1459
        prediction_df.to_csv('bdint/kaggle/submission.csv', index=False)


    def k_fold_validation(self, train_df, model, k=10):
    
        target = train_df[["SalePrice"]]
        features = train_df.drop(columns=["SalePrice"])
        target.columns = ["SalePrice"]

        
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        rmse_values = []

        for train_index, test_index in kf.split(features):
            x_train, x_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]


            # Example: Use a linear regression model
            model.learn(x_train, y_train)

            # Calculate RMSE
            predictions = model.predict(x_test)

               
            
            rmse = self.rmse( predictions["SalePrice"], y_test["SalePrice"])
            rmse_values.append(rmse)

        # Calculate the mean RMSE
        mean_rmse = sum(rmse_values) / len(rmse_values)

        return mean_rmse
    
    def rmse(self, test, prediciton):
        s = []
        for idx, t in enumerate(test):
            a = t
            b = prediciton.iloc[idx]
            sqr = (a-b)**2
            s.append(sqr)
        return np.sqrt(np.sum(s)/len(s))

