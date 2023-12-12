from bdint.data import (
    get_test_df,
    get_train_df,
    k_fold_validation,
    make_kaggle_submission_file,
)
from bdint.models import RandomForest
from bdint.models.utils import ohe

train_df, validation_df = get_train_df()
test_df = get_test_df()

print("Train Set Size:", len(train_df))
print("Validation Set Size:", len(validation_df))
print("Test Set Size:", len(test_df))

# create Model
model = RandomForest()

train_df = train_df[["Alley", "ScreenPorch", "SalePrice"]]
test_df = train_df[["Alley", "ScreenPorch", "SalePrice"]]

# numerical
# train_df = preprocess_for_numerical_model(train_df)
# test_df = preprocess_for_numerical_model(test_df)

# ohe
train_df_ohe, test_df_ohe = ohe(train_df, test_df)

print(train_df_ohe)

# Check Performance of model using k validation
rmse = k_fold_validation(train_df=train_df_ohe, model=model, k=10)
print("RMSE:", rmse)

# predict test set
prediction = model.predict(x_test_df=test_df_ohe)
# create kaggle submission file
make_kaggle_submission_file(prediction, test_df)
