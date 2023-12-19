from bdint.models import RandomForest
from bdint.models import CatBoost
from bdint.data import k_fold_validation
import time
import matplotlib.pyplot as plt
from bdint.models.utils import ohe


def RF_test_estimators(df):
    # write header in anal file. MAybe usefull later when plotting
    with open("bdint/anal/random_tree/estimators.txt", "w") as file:
        file.write("Estimators\tRMSE\tTime\n")

    max_it = 100

    for est in range(1, max_it):
        print(f"Iteration {est}/{max_it-1}\t", end="")
        estimator = est * 20

        time_start = time.time()
        model = RandomForest(n_estimators=estimator, random_state=42)
        rmse = k_fold_validation(df, model=model)
        time_to_calc = time.time() - time_start

        print(f"{rmse}\t{time_to_calc}s")

        with open("bdint/anal/random_tree/estimators.txt", "a") as file:
            file.write(f"{estimator}\t{rmse}\t{time_to_calc}\n")


def CB_one_feature_test(df):
    numerical_features = df.select_dtypes(include="number").columns.tolist()

    # baseline
    model = CatBoost()
    train_df_ohe, test_df_ohe = ohe(df, df)
    baseline_rmse = k_fold_validation(train_df=train_df_ohe, model=model)
    with open("bdint/anal/cat_boost/one_feature.txt", "w") as file:
        file.write(f"ALL\t{baseline_rmse}\n")

    l = len(numerical_features)
    for idx, num in enumerate(numerical_features):
        if num == "Id":
            continue
        print(f"Test {idx}/{l}")
        df_copy = df.copy()
        df_copy = df_copy.drop(num, axis=1)

        model = CatBoost()
        train_df_ohe, test_df_ohe = ohe(df_copy, df_copy)
        rmse = k_fold_validation(train_df=train_df_ohe, model=model)

        correlation = df[num].corr(df["SalePrice"])
        dif = rmse - baseline_rmse

        with open("bdint/anal/cat_boost/one_feature.txt", "a") as file:
            file.write(f"{num}\t{rmse}\t{correlation}\t{dif}\n")


def plot_one_with_time(path, title):
    x_values = []
    y_values = []
    time_values = []
    x_label = ""
    y_label = ""

    # Read data from the file
    with open(path, "r") as file:
        for idx, line in enumerate(file):
            values = line.strip().split("\t")
            if idx == 0:
                x_label = values[0]
                y_label = values[1]

                continue

            x_values.append(float(values[0]))
            y_values.append(float(values[1]))
            time_values.append(float(values[2]))

    # Create a figure and axis
    fig, ax1 = plt.subplots()

    # Plot the first data series
    color = "tab:red"
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label, color=color)
    ax1.plot(x_values, y_values, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    # Create a second y-axis
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Time [s]", color=color)
    ax2.plot(x_values, time_values, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(title)

    plt.savefig(f'bdint/vizualization/{title.replace(" ", "_")}.png')
    plt.show()
