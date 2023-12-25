import numpy as np
import torch
import torch.nn as nn

from bdint.models.basemodel import BaseModel
from bdint.models.utils import OHE, preprocess_for_categorical_model


class NN(nn.Module, BaseModel):
    def _preprocess(self, df, predicting=False) -> torch.Tensor:
        df = preprocess_for_categorical_model(df)
        ohe = self.ohe.ohe(df, use_category=True, categorical_columns_parameter=self.categorical_columns)

        if isinstance(ohe, tuple):
            ohe, self.categorical_columns = ohe
        matrix = ohe.values.astype(np.float32)
        if matrix.shape[1] != self.hidden1.in_features:
            if predicting:
                raise ValueError(
                    f"Model was trained with {self.hidden1.in_features} features, but {matrix.shape[1]} were given"
                )
            self.hidden1 = nn.Linear(matrix.shape[1], self.hidden1.out_features)
        return torch.from_numpy(matrix)

    def __init__(
        self, hidden_size1, hidden_size2, hidden_size3, epochs=500, lr=0.5, weight_decay=0.1, categorical_columns=None
    ):
        super().__init__()
        self.hidden1 = nn.Linear(1, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.hidden3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.output = nn.Linear(hidden_size3, 1)
        self.relu = nn.ReLU()
        self.ohe = OHE()
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.categorical_columns = categorical_columns

    def forward(self, x):
        out = self.hidden1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.hidden2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.hidden3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.output(out)
        return out

    def learn(self, x_train_df, y_train_df):
        self.__init__(
            self.hidden1.out_features,
            self.hidden2.out_features,
            self.hidden3.out_features,
            self.epochs,
            self.lr,
            self.weight_decay,
        )
        x_train_tensor = self._preprocess(x_train_df)
        y_train_tensor = torch.from_numpy(y_train_df.values.astype(np.float32))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(self.epochs):
            y_pred = self(x_train_tensor)
            loss = criterion(y_pred, y_train_tensor)
            print(f"Epoch: {epoch}, Loss: {np.sqrt(loss.item())}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, x_test_df):
        x_test_tensor = self._preprocess(x_test_df, predicting=True)
        return self.forward(x_test_tensor).detach().numpy()
