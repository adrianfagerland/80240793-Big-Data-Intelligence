import numpy as np

from bdint.data import (
    get_test_df,
    get_train_df,
    k_fold_validation,
    make_kaggle_submission_file,
)
from bdint.models import NN, BaseModel, CatBoost, KernelRidgeRegression, RandomForest

train_df = get_train_df()
test_df = get_test_df()

print("Train Set Size:", len(train_df))
print("Test Set Size:", len(test_df))


models: list[BaseModel] = [
    RandomForest(),
    CatBoost(),
    NN(hidden_size1=1028, hidden_size2=512, hidden_size3=256, epochs=10),
    KernelRidgeRegression(train=train_df, test=test_df, skewness_threshold=1),
]

target = train_df[["SalePrice"]]
features = train_df.drop(columns=["SalePrice"])
target.columns = ["SalePrice"]
predictions = []
for model in models:
    score = k_fold_validation(train_df, model=model, k=5)
    print(f"k-fold Validation for model {model}: {score}")
    model.learn(features, target)
    prediction = model.predict(test_df)
    predictions.append(prediction)

# aggregate predictions by weights rf: 0.1 catboost: 0.3 nn: 0.3 krr: 0.3
prediction = np.average(predictions, axis=0, weights=[0.1, 0.3, 0.3, 0.3])
make_kaggle_submission_file(prediction, test_df)
