import pandas as pd
import numpy as np

from bdint.data import (
    get_test_df,
    get_train_df,
    k_fold_validation,
    make_kaggle_submission_file,
)
from bdint.features import heatmap, categorical_boxplot
from bdint.models import CatBoost, RandomForest, KernelRidgeRegression
import matplotlib.pyplot as plt

train_df = get_train_df()
test_df = get_test_df()

print("Train Set Size:", len(train_df))
print("Test Set Size:", len(test_df))


model = KernelRidgeRegression(train=train_df, test=test_df)

# train model
score = k_fold_validation(model=model, k=5)
print(f"k-fold Validation: {score}")

# predict test set
model.learn()
prediction = np.expm1(model.predict())
plt.hist(prediction, bins=20, color="blue", edgecolor="black")
plt.show()
# create kaggle submission file
make_kaggle_submission_file(prediction, test_df)
