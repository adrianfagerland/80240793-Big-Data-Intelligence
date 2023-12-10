from bdint.data import (
    get_test_df,
    get_train_df,
    k_fold_validation,
    make_kaggle_submission_file,
)
from bdint.models import AveragePredicter, RandomForest

train_df, validation_df = get_train_df()
test_df = get_test_df()

print("Train Set Size:", len(train_df))
print("Validation Set Size:", len(validation_df))
print("Test Set Size:", len(test_df))

# create Model
model = RandomForest()

# Check Performance of model using k validation
rmse = k_fold_validation(train_df=train_df, model=model, k=10)
print("RMSE:", rmse)

# predict test set
prediction = model.predict(x_test_df=test_df)
# create kaggle submission file
make_kaggle_submission_file(prediction, test_df)
