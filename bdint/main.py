import pandas as pd

from bdint.data import (
    get_test_df,
    get_train_df,
    k_fold_validation,
    make_kaggle_submission_file,
)
from bdint.models import CatBoost, KernelRidge, RandomForest
from bdint.preprocessing import preprocessor

train_df = get_train_df()
test_df = get_test_df()

print("Train Set Size:", len(train_df))
print("Test Set Size:", len(test_df))

train, test = preprocessor(train_df, test_df)

# create Model
# model = RandomForest(n_estimators=100, random_state=42)
# set jobtype to cpu
# model = CatBoost(early_stopping_rounds=2000, iterations=10000)
# model = CatBoost(early_stopping_rounds=2000, iterations=15000)
model = RandomForest()
# numerical
# train_df = preprocess_for_numerical_model(train_df)
# test_df = preprocess_for_numerical_model(test_df)

# Check Performance of model using k validation
# rmse = k_fold_validation(train_df=train_df, model=model)
# print("RMSE:", rmse)


# train model
model.learn(
    x_train_df=train_df.drop(columns=["SalePrice"], inplace=False),
    y_train_df=pd.DataFrame(train_df["SalePrice"]),
)
# predict test set
prediction = model.predict(x_test_df=test_df)
# create kaggle submission file
make_kaggle_submission_file(prediction, test_df)
