from bdint.data import get_test_df, get_train_df
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# train_df = get_train_df()
# test_df = get_test_df()


def plot_saleprice_hist(df):
    # histogramm of prices. Motivation: "get an overview over the prices. We need this for better interpretation of the rmse we get"
    data = df["SalePrice"]

    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, color="#431C53", edgecolor="black")

    # Customize the plot
    plt.title("Histogram of SalePrice")
    plt.xlabel("SalePrice")
    plt.ylabel("Frequency")

    # Show the plot
    plt.savefig("bdint/vizualization/saleprice_historgram.png")
    plt.show()


def heatmap(df):
    df = df.copy()
    numerical_df = df.select_dtypes(include=["number"])
    print("Num numerical", numerical_df.shape[1])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numerical_df.corr(), annot=False)

    plt.savefig("bdint/vizualization/heatmap.png")
    plt.show()


def numerical_scatter_regression(df):
    df = df.copy()
    numerical_df = df.select_dtypes(include=["number"])
    for column in numerical_df.columns:
        correlation_coefficient = df[column].corr(df["SalePrice"])

        plt.figure(figsize=(10, 6))
        sns.regplot(x=column, y="SalePrice", data=df, line_kws={"color": "#431C53"})
        plt.title(f"SalePrice vs {column}, Corr = {round(correlation_coefficient,2)}")
        plt.xlabel(column)
        plt.ylabel("SalePrice")
        plt.savefig(
            f"bdint/vizualization/numerical_regression/scatter_plot_{int(np.abs(round(correlation_coefficient,2)*100))}_{column}.png"
        )
        plt.close()


def categorical_boxplot(df):
    df = df.copy()
    categorical_df = df.select_dtypes(include="object")

    for column in categorical_df.columns:
        plt.figure(figsize=(10, 6))

        # Box plot
        sns.set_palette("pastel")
        sns.boxplot(x=column, y="SalePrice", data=df)

        # Set plot title and labels
        plt.title(f"Box Plot for {column}")
        plt.xlabel(column)
        plt.ylabel("SalePrice")

        # Save the plot
        plt.savefig(f"bdint/vizualization/categorical_box_plot/box_plot_{column}.png")
        plt.close()


def analyse_numerical(df):
    numerical_features = df.select_dtypes(include=["number"])

    for column in numerical_features.columns:
        column_stats = df[column].describe()

        median_value = df[column].median()
        std_dev = df[column].std()

        missing_percentage = df[column].isna().mean() * 100

        correlation_sale_price = df[column].corr(df["SalePrice"])

        print_console = False
        if print_console:
            print(
                f"Feature: {column}, Min: {column_stats['min']}, Max: {column_stats['max']}, Mean: {column_stats['mean']}, Median: {median_value}, Std Dev: {std_dev}, Missing %: {missing_percentage:.2f}%"
            )

        print_for_latex = True
        if print_for_latex:
            color = "black"
            if abs(correlation_sale_price) < 0.2:
                color = "redcorr"
            elif abs(correlation_sale_price) < 0.4:
                color = "mediumredcorr"
            elif abs(correlation_sale_price) < 0.6:
                color = "okcorr"
            else:
                color = "greencorr"

            cor_col = "\\textcolor{%s}{%s}\\color{black}" % (
                color,
                (round(correlation_sale_price, 2)),
            )
            print(
                f"{column} & {column_stats['min']} & {column_stats['max']} &  {column_stats['mean']:.2f} & {median_value:.2f} & {std_dev:.2f} &{missing_percentage:.2f}\% & {cor_col} \\\\"
            )


def analyse_categorical(df):
    categorical_features = df.select_dtypes(include=["object"])

    for column in categorical_features.columns:
        num_categories = df[column].nunique()

        category_counts = df[column].value_counts()
        category_percentages = category_counts / len(df) * 100

        top_categories = category_percentages.head(2)

        missing_percentage = df[column].isna().mean() * 100

        print_console = False
        if print_console:
            print()
            print(
                f"Feature: {column}, Distinct Categories: {num_categories}, Top 2 Categories: {', '.join(top_categories.index)} ({top_categories.iloc[0]:.2f}%, {top_categories.iloc[1]:.2f}%)"
            )

        print_for_latex = True
        if print_for_latex:
            print(
                f"{column} & {num_categories} & {top_categories.index[0]} & {top_categories.iloc[0]:.2f}\% & {top_categories.index[1]} & {top_categories.iloc[1]:.2f}\% & {missing_percentage:.2f}\% \\\\"
            )


# plot_saleprice_hist(train_df)
# analyse_numerical(train_df)
