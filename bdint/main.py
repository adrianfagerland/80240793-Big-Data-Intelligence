import pandas as pd

from bdint.data import (
    get_test_df,
    get_train_df,
    k_fold_validation,
    make_kaggle_submission_file,
)
from bdint.models import RandomForest
from bdint.models.utils import ohe

train_df = get_train_df()
test_df = get_test_df()

print("Train Set Size:", len(train_df))
print("Test Set Size:", len(test_df))

# create Model
model = RandomForest(n_estimators=100, random_state=42)

# numerical
# train_df = preprocess_for_numerical_model(train_df)
# test_df = preprocess_for_numerical_model(test_df)

# ohe
train_df_ohe, test_df_ohe = ohe(train_df, test_df)

# Check Performance of model using k validation
rmse = k_fold_validation(train_df=train_df_ohe, model=model)
print("RMSE:", rmse)


# train model
model.learn(
    x_train_df=train_df_ohe.drop(columns=["SalePrice"], inplace=False),
    y_train_df=pd.DataFrame(train_df_ohe["SalePrice"]),
)
# predict test set
prediction = model.predict(x_test_df=test_df_ohe)
# create kaggle submission file
make_kaggle_submission_file(prediction, test_df_ohe)
