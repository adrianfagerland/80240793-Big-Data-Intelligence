import pandas as pd

class AverageModel:
    def __init__(self):
        self.mean = None  

    def learn(self, x_train_df, y_train_df):
       self.mean = y_train_df['SalePrice'].mean()

    def predict(self, x_test_df):
       
        if self.mean is None:
            raise ValueError("Model has not been trained. Call 'learn' method first.")
        
        
        result_df = pd.DataFrame(columns=['Id', 'SalePrice'])
        for index, row in x_test_df.iterrows():
            id = row['Id']
            prediction = self.mean
            result_df.loc[len(result_df)] = {"Id": id, "SalePrice": prediction}


        return result_df
