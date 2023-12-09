
from data import Data
from models.average import AverageModel

train_df, validation_df = Data().get_train_df()
test_df = Data().get_test_df()

print("Train Set Size:", len(train_df))
print("Validation Set Size:", len(validation_df))
print("Test Set Size:", len(test_df))

#create Model
model = AverageModel()

#Check Performance of model using k validation
rmse = Data().k_fold_validation(train_df=train_df, model=model, k = 10)
print("RMSE:" , rmse)

#predict test set
prediction = model.predict(x_test_df=test_df)
#create kaggle submission file
Data().make_kaggle_submission_file(prediction)