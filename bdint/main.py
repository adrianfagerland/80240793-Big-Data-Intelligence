import pandas as pd
import torch

from bdint.data import (
    get_test_df,
    get_train_df,
    k_fold_validation,
    make_kaggle_submission_file,
)
from bdint.models import NN, CatBoost, RandomForest

train_df = get_train_df()
test_df = get_test_df()

print("Train Set Size:", len(train_df))
print("Test Set Size:", len(test_df))

# create Model
# model = RandomForest(n_estimators=100, random_state=42)
# set jobtype to cpu
# model = CatBoost(early_stopping_rounds=2000, iterations=10000)
# model = RandomForest(n_estimators=100, random_state=42)
model = NN(hidden_size1=1028, hidden_size2=512, hidden_size3=256)

# train model
model.learn(
    x_train_df=train_df.drop(columns=["SalePrice"], inplace=False),
    y_train_df=pd.DataFrame(train_df["SalePrice"]),
)
# predict test set
prediction = model.predict(x_test_df=test_df)
if isinstance(prediction, torch.Tensor):
    prediction = prediction.detach().numpy()
# create kaggle submission file
make_kaggle_submission_file(prediction, test_df)
