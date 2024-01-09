import matplotlib.pyplot as plt
import pandas as pd
import torch

from bdint.data import get_test_df, get_train_df, make_kaggle_submission_file
from bdint.models import NN, CatBoost

train_df = get_train_df()
test_df = get_test_df()

print("Train Set Size:", len(train_df))
print("Test Set Size:", len(test_df))


model = CatBoost()

# train model
model.learn(
    x_train_df=train_df.drop(columns=["SalePrice"], inplace=False),
    y_train_df=pd.DataFrame(train_df["SalePrice"]),
)
# predict test set
prediction = model.predict(x_test_df=test_df)

plt.hist(prediction, bins=20, color="blue", edgecolor="black")
plt.show()
if isinstance(prediction, torch.Tensor):
    prediction = prediction.detach().numpy()
# create kaggle submission file
make_kaggle_submission_file(prediction, test_df)
