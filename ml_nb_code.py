import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib.colors import ListedColormap
import ipywidgets as widgets
from graphviz import Digraph
from IPython.display import display


def _split_feature(fd1, fd2, fd3, check, sl1, sl2, sl3, df):
    f, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    ax_ = ax.twinx()
    df1 = df.loc[df[fd1] > sl1, :]
    df0 = df.loc[df[fd1] <= sl1, :]
    ms = 30
    ax.plot(df0.loc[df0["safe"], fd1], df0.loc[df0["safe"], fd2], "go", ms=ms)
    ax.plot(df0.loc[~df0["safe"], fd1], df0.loc[~df0["safe"], fd2], "ro", ms=ms)
    ax_.plot(df1.loc[df1["safe"], fd1], df1.loc[df1["safe"], fd3], "go", ms=ms)
    ax_.plot(df1.loc[~df1["safe"], fd1], df1.loc[~df1["safe"], fd3], "ro", ms=ms)
    ax.plot([], [], "go", ms=10, label="safe")
    ax.plot([], [], "ro", ms=10, label="unsafe")
    ax.axvline(
        x=sl1, color="gray", linestyle="--"
    )  # , label=f"Split at {fd1}={sl1:.1f}")
    ax.set_xlabel(fd1)
    # ax.set_yticks([])
    ax.set_ylabel(fd2)
    ax_.set_ylabel(fd3)
    if check:
        xlim = ax.get_xlim()
        ax.set_xlim(xlim)
        if fd2 == "material_type":
            sl2 = 0.5
        if fd3 == "material_type":
            sl3 = 0.5
        ax.plot([xlim[0], sl1], [sl2, sl2], "--", color="gray")
        ax_.plot([sl1, xlim[1]], [sl3, sl3], "--", color="gray")
    else:
        ylim0 = ax.get_ylim()
        ylim1 = ax_.get_ylim()
        ax.set_ylim([np.min([ylim0[0], ylim1[0]]), np.max([ylim0[-1], ylim1[-1]])])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1))

    ax1.set_xlim([-0.3, 2.5])
    ax1.set_ylim([-3.5, 3.5])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis("off")
    bb = dict(facecolor="white", edgecolor="black")
    ax1.plot([0, 1], [0, 2], "k-")
    ax1.plot([0, 1], [0, -2], "k-")
    if check:
        ax1.plot([1, 2], [2, 3], "k-")
        ax1.plot([1, 2], [2, 1], "k-")
        ax1.plot([1, 2], [-2, -3], "k-")
        ax1.plot([1, 2], [-2, -1], "k-")

    if not check:
        c2 = [0.5, 0.5, 0.5]
        c1 = "k"
    else:
        c1 = [0.5, 0.5, 0.5]
        c2 = "k"
    ax1.text(0, 0, f"{fd1}\n>{sl1:.1f}", ha="center", bbox=bb, color=c1)
    if check:
        if fd2 == "material_type":
            txt = f"{fd2}\n is steel"
        else:
            txt = f"{fd2}\n>{sl2:.1f}"
        ax1.text(1, -2, txt, ha="center", bbox=bb, color=c2)
        if fd3 == "material_type":
            txt = f"{fd3}\n is steel"
        else:
            txt = f"{fd3}\n>{sl3:.1f}"
        ax1.text(1, 2, txt, ha="center", bbox=bb, color=c2)

    if fd2 == "material_type":
        df00 = df0.loc[df0[fd2] == "Steel", :]
        df01 = df0.loc[df0[fd2] == "Concrete", :]
    else:
        df00 = df0.loc[df0[fd2] > sl2, :]
        df01 = df0.loc[df0[fd2] <= sl2, :]
    if fd3 == "material_type":
        df10 = df1.loc[df1[fd3] == "Steel", :]
        df11 = df1.loc[df1[fd3] != "Steel", :]
    else:
        df10 = df1.loc[df1[fd3] > sl3, :]
        df11 = df1.loc[df1[fd3] <= sl3, :]

    if check:
        ys = [-1, -3, 3, 1]
        dfs = [df00, df01, df10, df11]
    else:
        ys = [-2, 2]
        dfs = [df0, df1]
    for y, dfi in zip(ys, dfs):
        if dfi.shape[0] == 0:
            s = 0
            us = 0
        else:
            s = dfi["safe"].sum()
            us = dfi.shape[0] - s
        if check:
            ax1.text(2, y, f"{s:d} Safe\n{us:d} Unsafe", bbox=bb, ha="center", color=c2)
        else:
            ax1.text(1, y, f"{s:d} Safe\n{us:d} Unsafe", bbox=bb, ha="center", color=c2)

    bb = dict(facecolor="white", edgecolor="blue", boxstyle="round")
    ax1.text(0.5, 1, "True", ha="center", bbox=bb, style="italic", color="b")
    ax1.text(0.5, -1, "False", ha="center", bbox=bb, style="italic", color="b")
    if check:
        ax1.text(1.5, 2.5, "True", ha="center", bbox=bb, style="italic", color="b")
        ax1.text(1.5, 1.5, "False", ha="center", bbox=bb, style="italic", color="b")
        ax1.text(1.5, -1.5, "True", ha="center", bbox=bb, style="italic", color="b")
        ax1.text(1.5, -2.5, "False", ha="center", bbox=bb, style="italic", color="b")

    if fd2 == fd3:
        y0 = np.min([axi.get_ylim()[0] for axi in [ax, ax_]])
        y1 = np.max([axi.get_ylim()[1] for axi in [ax, ax_]])
        [axi.set_ylim([y0, y1]) for axi in [ax, ax_]]

    plt.show()


def get_safe_unsafe_data():
    # Create the dataframe
    data = [
        {"load_capacity": 50, "material_type": "Concrete", "age": 10, "safe": False},
        {"load_capacity": 30, "material_type": "Concrete", "age": 5, "safe": True},
        {"load_capacity": 70, "material_type": "Concrete", "age": 25, "safe": False},
        {"load_capacity": 70, "material_type": "Steel", "age": 35, "safe": False},
        {"load_capacity": 60, "material_type": "Steel", "age": 15, "safe": True},
        {"load_capacity": 50, "material_type": "Steel", "age": 8, "safe": True},
        {"load_capacity": 35, "material_type": "Steel", "age": 3, "safe": True},
    ]
    df = pd.DataFrame(data)

    return df


def decision_tree_old():
    df = get_safe_unsafe_data()

    # Create the interactive widgets
    fd1 = widgets.widgets.Dropdown(
        value="load_capacity", options=["load_capacity", "age"], description="Feature:"
    )
    fd2 = widgets.widgets.Dropdown(
        value="age",
        options=["load_capacity", "age", "material_type"],
        description="Feature:",
    )
    fd3 = widgets.widgets.Dropdown(
        value="age",
        options=["load_capacity", "age", "material_type"],
        description="Feature:",
    )
    check = widgets.widgets.Checkbox(value=False, description="lock root node")

    sl1 = widgets.widgets.FloatSlider(
        value=50,
        min=min(df["load_capacity"].min(), df["age"].min()) - 1,
        max=max(df["load_capacity"].max(), df["age"].max()) + 1,
        step=1,
        description="Split value:",
        continuous_update=False,
    )
    sl2 = widgets.widgets.FloatSlider(
        value=10,
        min=min(df["load_capacity"].min(), df["age"].min()) - 1,
        max=max(df["load_capacity"].max(), df["age"].max()) + 1,
        step=1,
        description="Split value:",
        continuous_update=False,
    )
    sl3 = widgets.widgets.FloatSlider(
        value=10,
        min=min(df["load_capacity"].min(), df["age"].min()) - 1,
        max=max(df["load_capacity"].max(), df["age"].max()) + 1,
        step=1,
        description="Split value:",
        continuous_update=False,
    )
    # sl1.value=53
    # fd2.value='material_type'
    # check.value=True
    io = widgets.interactive_output(
        _split_feature,
        {
            "fd1": fd1,
            "fd2": fd2,
            "fd3": fd3,
            "check": check,
            "sl1": sl1,
            "sl2": sl2,
            "sl3": sl3,
            "df": widgets.fixed(df),
        },
    )
    return widgets.VBox(
        [
            widgets.HBox([fd1, sl1, check]),
            io,
            widgets.HBox([widgets.VBox([fd2, sl2]), widgets.VBox([fd3, sl3])]),
        ]
    )


def scaling_plot(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, scale: float):
    x2 = x2 * scale

    # Plot the data
    fig = plt.figure()
    plt.scatter(x1, y, c="r", label="X1")
    plt.scatter(x2, y, c="g", label=f"X2 * {scale:.3f}")
    plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    plt.xlabel("Features: X1, X2")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()


def get_fs_data(w1=-0.3, w2=0.4):
    x1_orig = np.random.rand(100)
    x2_orig = np.random.rand(100)

    # w1, w2 = -0.3, 0.4
    y = w1 * x1_orig + w2 * x2_orig

    x1 = x1_orig * 0.1
    x2 = x2_orig * 10

    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    return df


def feature_scaling():
    df = get_fs_data(w1=-0.5, w2=0.6)

    widgets.interact(
        scaling_plot,
        x1=widgets.fixed(df.x1),
        x2=widgets.fixed(df.x2),
        y=widgets.fixed(df.y),
        scale=widgets.FloatLogSlider(
            1, base=10, min=-3, max=3, step=1, description="X2 Scale"
        ),
    )


def load_iris_df():
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    df = pd.DataFrame(data=X, columns=feature_names)
    df["target"] = [iris.target_names[cur_ix] for cur_ix in iris.target]
    df = df.sample(frac=1)

    return df, feature_names


def plot_decision_boundaries_iris(
    clf,
    iris_df: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    fixed_feature_1: str,
    fixed_value_1: float,
    fixed_feature_2: str,
    fixed_value_2,
    feature_names: list[str],
    ax: plt.Axes,
    x_lim: tuple[float, float] = None,
    y_lim: tuple[float, float] = None,
):
    target_classes = list(clf.classes_)

    x_min, x_max = (
        x_lim
        if x_lim is not None
        else (iris_df[feature_1].min(), iris_df[feature_1].max())
    )
    y_min, y_max = (
        y_lim
        if y_lim is not None
        else (iris_df[feature_2].min(), iris_df[feature_2].max())
    )

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 1000),
        np.linspace(y_min, y_max, 1000),
    )

    # Create a grid of points in the feature_idx1-feature_idx2 plane
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Create a full grid in 3D by adding the fixed value for the third dimension
    grid_df = pd.DataFrame(
        np.zeros((grid.shape[0], 4)),
        columns=[feature_1, feature_2, fixed_feature_1, fixed_feature_2],
    )
    grid_df.loc[:, feature_1] = grid[:, 0]
    grid_df.loc[:, feature_2] = grid[:, 1]
    grid_df.loc[:, fixed_feature_1] = fixed_value_1
    grid_df.loc[:, fixed_feature_2] = fixed_value_2

    # Predict the class for each point in the grid
    Z = clf.predict(grid_df[feature_names]).reshape(xx.shape)

    Z_int = np.array([target_classes.index(i) for i in Z.ravel()]).reshape(xx.shape)

    # Plot the decision boundaries
    ax.contourf(xx, yy, Z_int, alpha=0.3)
    ax.scatter(
        iris_df.loc[:, feature_1],
        iris_df.loc[:, feature_2],
        c=iris_df.loc[:, "target_encoded"],
        edgecolors="k",
        marker="o",
    )
    ax.set_xlabel(f"Feature {feature_1}")
    ax.set_ylabel(f"Feature {feature_2}")
    ax.set_title(
        f"Fixed Feature {fixed_feature_1} = {fixed_value_1:.2f}, {fixed_feature_2} = {fixed_value_2:.2f}"
    )


def feature_scaling_example():
    df = get_fs_data()

    df["x1_norm"] = (df.x1 - df.x1.mean()) / df.x1.std()
    df["x2_norm"] = (df.x2 - df.x2.mean()) / df.x2.std()

    # Plot the data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(df.x1, df.y, c="r", label="X1")
    ax1.scatter(df.x2, df.y, c="g", label="X2")
    ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_xlabel("Features: X1, X2")
    ax1.set_ylabel("Target")
    ax1.legend()

    ax2.scatter(df.x1_norm, df.y, c="r", label="X1")
    ax2.scatter(df.x2_norm, df.y, c="g", label="X2")
    ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_xlabel("Normalised Features: X1, X2")
    ax2.set_ylabel("Target")

    fig.tight_layout()


def linear_regression_fitting_example():
    np.random.seed(5)

    # Create toggle buttons
    toggle1 = widgets.Checkbox(value=False, description="Show fit 1")

    toggle2 = widgets.Checkbox(value=False, description="Show fit 2")

    toggle3 = widgets.Checkbox(value=False, description="Show fit 3")
    hbox = widgets.HBox([toggle1, toggle2, toggle3])
    # display(toggle1, toggle2, toggle3)
    display(hbox)

    output = widgets.Output()

    @output.capture(clear_output=True, wait=True)
    def gen_plot(*args):
        # import datetime
        # print(f"-------------------------")
        # print(f"{datetime.datetime.now()}")
        # print(f"{toggle1.value}")
        # print(f"{toggle2.value}")
        # print(f"{toggle3.value}")
        df = get_fs_data(w1=-0.7)
        df = df[::5]
        x = np.linspace(df.x1.min(), df.x1.max(), 1000)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df.x1, df.y, c="r", label="Data")

        if toggle1.value:
            ax.axhline(df.y.mean(), c="purple", label="Model 1")

        if toggle2.value:
            poly = PolynomialFeatures(degree=10)
            X = poly.fit_transform(df.x1.values[:, None])

            poly_model = LinearRegression().fit(X, df.y)
            ax.plot(
                x,
                poly_model.predict(poly.fit_transform(x[:, None])),
                c="g",
                label="Model 1",
            )

        if toggle3.value:
            model = LinearRegression().fit(df.x1.values[:, None], df.y.values)
            ax.plot(x, model.predict(x[:, None]), c="b", label="Model 3")

        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
        ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

        ax.legend()
        plt.show()

    toggle1.observe(gen_plot, names="value")
    toggle2.observe(gen_plot, names="value")
    toggle3.observe(gen_plot, names="value")

    # Display the output widget
    display(output)

    gen_plot()


def get_heart_df(features: list[str] = None, drop_na: bool = True):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]
    df = pd.read_csv(url, header=None, names=columns, na_values="?")
    if drop_na:
        df = df.dropna()  # Drop rows with missing values

    categorial_mapping = {
        "cp": {
            1: "Typical",
            2: "Atypical",
            3: "Non_anginal",
            4: "Asymptomatic",
        },
        "restecg": {
            0: "Normal",
            1: "ST-T_abnormality",
            2: "Left_ventricular_hypertrophy",
        },
        "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
        "thal": {3: "Normal", 6: "Fixed_defect", 7: "Reversable_defect"},
        "sex": {1: "Male", 0: "Female"},
    }
    for cur_key, cur_mapping in categorial_mapping.items():
        df[cur_key] = df[cur_key].map(cur_mapping)

    if features is not None:
        cols = features + ["target"]
        df = df[cols]

    # Make binary 0 - no presence, (1, 2, 3, 4) - presence
    df["target"] = df["target"].apply(lambda x: "Presence" if x == 0 else "No Presense")

    return df


def get_nan_example():
    df = pd.DataFrame(
        data=np.random.random(30).reshape(10, 3),
        columns=["Feature1", "Feature2", "Feature3"],
    )
    nan_indices_1 = np.random.choice(df.index, 2, replace=False)
    df.loc[nan_indices_1, "Feature1"] = np.nan

    nan_indices_2 = np.random.choice(df.index, 1, replace=False)
    df.loc[nan_indices_2, "Feature2"] = np.nan

    nan_indices_3 = np.random.choice(df.index, 1, replace=False)
    df.loc[nan_indices_3, "Feature3"] = np.nan

    df["Target"] = [
        "Safe",
        "Unsafe",
        "Unsafe",
        "Safe",
        "Unsafe",
        "Safe",
        "Safe",
        "Unsafe",
        "Safe",
        "Unsafe",
    ]

    return df


def plot_single_decision_boundary_heart(
    heart_df,
    model,
    feature1,
    feature2,
    thal_key,
    features,
    thal_keys,
    ax,
    val_df=None,
    legend=False,
):
    m = heart_df[thal_key] == 1
    cur_heart_df = heart_df.copy()
    cur_heart_df = cur_heart_df.loc[m]

    x_min, x_max = heart_df[feature1].min() - 1.0, heart_df[feature1].max() + 1.0
    y_min, y_max = heart_df[feature2].min() - 1.0, heart_df[feature2].max() + 1.0

    # Generate the decision boundary
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000)
    )

    X = pd.DataFrame({feature1: xx.ravel(), feature2: yy.ravel()})
    X[thal_keys] = 0
    X[thal_key] = 1

    Z = model.predict(X[features])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    ax.contourf(
        xx, yy, Z, alpha=0.3, cmap=ListedColormap(("green", "red")), vmin=0, vmax=1
    )
    ax.scatter(
        cur_heart_df.loc[cur_heart_df.target_encoded == 1, [feature1]].values,
        cur_heart_df.loc[cur_heart_df.target_encoded == 1, [feature2]].values,
        c="red",
        label="Train - Presence",
        edgecolors="k",
    )
    ax.scatter(
        cur_heart_df.loc[cur_heart_df.target_encoded == 0, [feature1]].values,
        cur_heart_df.loc[cur_heart_df.target_encoded == 0, [feature2]].values,
        c="green",
        label="Train - No Presence",
        edgecolors="k",
    )
    if val_df is not None:
        cur_val_df = val_df.copy()
        cur_val_df = cur_val_df.loc[cur_val_df[thal_key] == 1]

        ax.scatter(
            cur_val_df.loc[val_df.target_encoded == 1, feature1].values,
            cur_val_df.loc[val_df.target_encoded == 1, feature2].values,
            c="red",
            marker="s",
            label="Val - Presence",
            edgecolors="k",
        )
        ax.scatter(
            cur_val_df.loc[val_df.target_encoded == 0, feature1].values,
            cur_val_df.loc[val_df.target_encoded == 0, feature2].values,
            c="green",
            marker="s",
            label="Val - No Presence",
            edgecolors="k",
        )
        if legend:
            ax.legend()
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title(f"Decision Boundary thal = {thal_key}, N = {m.sum()}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def plot_decision_boundary_heart(heart_df, clf, features, figsize=(14, 4), val_df=None):
    thal_keys = [cur_col for cur_col in heart_df.columns if cur_col.startswith("thal_")]

    # Plot decision boundary for each categorical value of 'thal'
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for ix, (cur_thal_key, cur_ax) in enumerate(zip(thal_keys, axs)):
        # fig = plt.figure(figsize=figsize)
        plot_single_decision_boundary_heart(
            heart_df,
            clf,
            "thalach",
            "oldpeak",
            cur_thal_key,
            features,
            thal_keys,
            cur_ax,
            val_df=val_df,
            legend=ix == 0,
        )
    fig.tight_layout()


def get_prepped_heart_df():
    heart_df = get_heart_df(features=["thalach", "oldpeak", "thal"])

    # Normalise the features
    numerical_features = ["thalach", "oldpeak"]
    std_scaler = StandardScaler()
    heart_df[numerical_features] = std_scaler.fit_transform(
        heart_df[numerical_features]
    )

    # Encode the categorical features
    categorical_features = ["thal"]
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(heart_df[categorical_features])
    heart_df[encoder.get_feature_names_out()] = encoded_data

    # Encode the labels
    label_encoder = LabelEncoder()
    heart_df["target_encoded"] = label_encoder.fit_transform(heart_df["target"])

    features_keys = numerical_features + list(encoder.get_feature_names_out())
    return heart_df, features_keys, label_encoder


def hyperparam_tuning_example():
    heart_df, feature_keys, _ = get_prepped_heart_df()

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        heart_df[feature_keys],
        heart_df["target_encoded"],
        test_size=0.2,
        random_state=42,
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)

    # Create widgets for hyperparameters
    style = {"description_width": "initial"}
    max_depth = widgets.IntSlider(
        value=10,
        min=1,
        max=10,
        step=1,
        description="Max Depth:",
        style=style,
        continuous_update=False,
    )
    min_samples_split = widgets.IntSlider(
        value=2,
        min=2,
        max=25,
        step=1,
        description="Min Samples Split:",
        style=style,
        continuous_update=False,
    )
    min_samples_leaf = widgets.IntSlider(
        value=1,
        min=1,
        max=25,
        step=1,
        description="Min Samples Leaf:",
        style=style,
        continuous_update=False,
    )
    hbox = widgets.HBox([max_depth, min_samples_split, min_samples_leaf])

    # Create an output widget to display the plot
    output = widgets.Output()

    # Define a function to update the plot
    @output.capture(clear_output=True, wait=True)
    def update_plot(*args):
        with output:
            # Create a Decision Tree classifier with the selected hyperparameters
            clf = DecisionTreeClassifier(
                max_depth=max_depth.value,
                min_samples_split=min_samples_split.value,
                min_samples_leaf=min_samples_leaf.value,
                random_state=42,
            )

            # Train the model on the training data
            clf.fit(train_df[feature_keys], train_df["target_encoded"])

            # Calculate accuracy
            val_acc = accuracy_score(
                val_df["target_encoded"], clf.predict(val_df[feature_keys])
            )
            train_acc = accuracy_score(
                train_df["target_encoded"], clf.predict(train_df[feature_keys])
            )
            print(f"Training Accuracy: {train_acc:.2f}")
            print(f"Validation Accuracy: {val_acc:.2f}")

            # Visualize the decision tree
            fig, ax = plt.subplots(figsize=(14, 4))
            plot_tree(
                clf,
                filled=True,
                ax=ax,
                feature_names=feature_keys,
                class_names=["No Presence", "Presence"],
                impurity=False,
            )

            plot_decision_boundary_heart(train_df, clf, feature_keys, val_df=val_df)
            plt.show()

    # Attach the update_plot function to the widgets
    max_depth.observe(update_plot, names="value")
    min_samples_split.observe(update_plot, names="value")
    min_samples_leaf.observe(update_plot, names="value")

    # Display the widgets and the output plot
    display(hbox, output)

    # Call the function initially to display the plot
    update_plot()


def decision_tree():
    class DecisionTreeNode:
        def __init__(
            self, feature=None, threshold=None, left=None, right=None, value=None
        ):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

        def is_leaf_node(self):
            return self.value is not None

    class DecisionTree:
        def __init__(self):
            self.root = None

        def build_tree_from_dict(self, tree_dict):
            """
            Recursively build the tree from a nested dictionary.
            """
            if "value" in tree_dict:
                return DecisionTreeNode(value=tree_dict["value"])

            node = DecisionTreeNode(
                feature=tree_dict["feature"], threshold=tree_dict["threshold"]
            )
            node.left = self.build_tree_from_dict(tree_dict["left"])
            node.right = self.build_tree_from_dict(tree_dict["right"])

            return node

        def build_tree(self, tree_dict):
            """
            Initialize the tree building process from the root.
            """
            self.root = self.build_tree_from_dict(tree_dict)

        def predict(self, X):
            """
            Predict the target value for each sample in X.
            """
            return np.array([self._traverse_tree(x, self.root) for x in X])

        def _traverse_tree(self, x, node):
            if node.is_leaf_node():
                return node.value
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

        def _get_node(self, x, node):
            if node.is_leaf_node():
                return node
            if x[node.feature] <= node.threshold:
                return self._get_node(x, node.left)
            else:
                return self._get_node(x, node.right)

    def visualize_tree(tree, features):
        target_lookup = {0: "Unsafe", 1: "Safe", -1: "No Data"}

        def add_nodes_edges(
            tree,
            features,
            dot=None,
        ):
            # Create Digraph object
            if dot is None:
                dot = Digraph()
                dot.node(
                    name=str(tree),
                    label=f"{features[tree.feature]} <= {tree.threshold}",
                )

            # Add nodes
            if tree.left:
                dot.node(
                    name=str(tree.left),
                    label=(
                        (
                            f"{features[tree.left.feature]} <= {tree.left.threshold}"
                            if tree.left.feature != 2
                            else features[2]
                        )
                        if not tree.left.is_leaf_node()
                        else f"Leaf Value: {target_lookup[tree.left.value]}"
                    ),
                )
                dot.edge(str(tree), str(tree.left), label="True")
                dot = add_nodes_edges(tree.left, features, dot=dot)

            if tree.right:
                dot.node(
                    name=str(tree.right),
                    label=(
                        (
                            f"{features[tree.right.feature]} <= {tree.right.threshold}"
                            if tree.right.feature != 2
                            else features[2]
                        )
                        if not tree.right.is_leaf_node()
                        else f"Leaf Value: {target_lookup[tree.right.value]}"
                    ),
                )
                dot.edge(str(tree), str(tree.right), label="False")
                dot = add_nodes_edges(tree.right, features, dot=dot)

            return dot

        # Add nodes recursively and create a dot object
        dot = add_nodes_edges(tree.root, features)
        return dot

    def plot(tree):

        fig, (ax_steel, ax_concrete) = plt.subplots(1, 2, figsize=(12, 6))

        ax_steel.set_title("Steel")
        ax_concrete.set_title("Concrete")

        ax_steel.set_xlim(20, 80)
        ax_steel.set_xlabel("Load Capacity")
        ax_concrete.set_xlim(20, 80)
        ax_concrete.set_xlabel("Load Capacity")

        ax_steel.set_ylim(0, 40)
        ax_steel.set_ylabel("Age")
        ax_concrete.set_ylim(0, 40)
        ax_concrete.set_ylabel("Age")

        if tree.root.feature == 0:
            ax_steel.axvline(tree.root.threshold, color="black", linestyle="--")
            ax_concrete.axvline(tree.root.threshold, color="black", linestyle="--")

            if tree.root.left.feature == 1:
                ax_steel.plot(
                    [0, tree.root.threshold],
                    [tree.root.left.threshold, tree.root.left.threshold],
                    color="black",
                    linestyle="--",
                )
                ax_concrete.plot(
                    [0, tree.root.threshold],
                    [tree.root.left.threshold, tree.root.left.threshold],
                    color="black",
                    linestyle="--",
                )
            elif tree.root.left.feature == 0:
                ax_steel.plot(
                    [tree.root.left.threshold, tree.root.left.threshold],
                    [0, tree.root.threshold],
                    color="black",
                    linestyle="--",
                )
                ax_concrete.plot(
                    [tree.root.left.threshold, tree.root.left.threshold],
                    [0, tree.root.threshold],
                    color="black",
                    linestyle="--",
                )

            if tree.root.right.feature == 1:
                ax_steel.plot(
                    [tree.root.threshold, 100],
                    [tree.root.right.threshold, tree.root.right.threshold],
                    color="black",
                    linestyle="--",
                )
                ax_concrete.plot(
                    [tree.root.threshold, 100],
                    [tree.root.right.threshold, tree.root.right.threshold],
                    color="black",
                    linestyle="--",
                )
            elif tree.root.right.feature == 0:
                ax_steel.plot(
                    [tree.root.right.threshold, tree.root.right.threshold],
                    [0, tree.root.threshold],
                    color="black",
                    linestyle="--",
                )
                ax_concrete.plot(
                    [tree.root.right.threshold, tree.root.right.threshold],
                    [0, tree.root.threshold],
                    color="black",
                    linestyle="--",
                )

        elif tree.root.feature == 1:
            ax_steel.axhline(tree.root.threshold, color="black", linestyle="--")
            ax_concrete.axhline(tree.root.threshold, color="black", linestyle="--")

            if tree.root.left.feature == 0:
                ax_steel.plot(
                    [tree.root.left.threshold, tree.root.left.threshold],
                    [0, tree.root.threshold],
                    color="black",
                    linestyle="--",
                )
                ax_concrete.plot(
                    [tree.root.left.threshold, tree.root.left.threshold],
                    [0, tree.root.threshold],
                    color="black",
                    linestyle="--",
                )
            elif tree.root.left.feature == 1:
                ax_steel.plot(
                    [0, tree.root.threshold],
                    [tree.root.left.threshold, tree.root.left.threshold],
                    color="black",
                    linestyle="--",
                )
                ax_concrete.plot(
                    [0, tree.root.threshold],
                    [tree.root.left.threshold, tree.root.left.threshold],
                    color="black",
                    linestyle="--",
                )

        elif tree.root.feature == 2:
            if tree.root.left.feature == 0:
                ax_concrete.axvline(
                    tree.root.left.threshold, color="black", linestyle="--"
                )
            elif tree.root.left.feature == 1:
                ax_concrete.axhline(
                    tree.root.left.threshold, color="black", linestyle="--"
                )

            if tree.root.right.feature == 0:
                ax_steel.axvline(
                    tree.root.right.threshold, color="black", linestyle="--"
                )
            elif tree.root.right.feature == 1:
                ax_steel.axhline(
                    tree.root.right.threshold, color="black", linestyle="--"
                )

        return fig, ax_steel, ax_concrete

    def run(
        df: pd.DataFrame,
        root_feature: str,
        root_treshold: float,
        left_feature: str,
        left_treshold: float,
        right_feature: str,
        right_treshold: float,
    ):

        # Yes, `is_concrete` is correct
        feature_names = ["load_capacity", "age", "is_concrete"]
        data = [
            (cur_row.load_capacity, cur_row.age, int(cur_row.material_type == "Steel"))
            for _, cur_row in df.iterrows()
        ]
        target_names = ["unsafe", "safe"]
        target = [int(cur_row.safe) for _, cur_row in df.iterrows()]

        tree_structure = {
            "feature": feature_names.index(root_feature),
            "threshold": root_treshold if root_feature != "is_concrete" else 0.5,
            "left": {
                "feature": feature_names.index(left_feature),
                "threshold": left_treshold if left_feature != "is_concrete" else 0.5,
                "left": {"value": -1},
                "right": {"value": -1},
            },
            "right": {
                "feature": feature_names.index(right_feature),
                "threshold": right_treshold if right_feature != "is_concrete" else 0.5,
                "left": {"value": -1},
                "right": {"value": -1},
            },
        }

        # Initialize the decision tree
        tree = DecisionTree()

        # Build the tree using the nested dictionary
        tree.build_tree(tree_structure)

        leaf_nodes = [
            (tree._get_node(cur_x, tree.root), cur_target)
            for cur_x, cur_target in zip(data, target)
        ]

        leaf_targets = {}
        for cur_node, cur_target in leaf_nodes:
            if cur_node not in leaf_targets:
                leaf_targets[cur_node] = []
            leaf_targets[cur_node].append(cur_target)

        for cur_node, cur_targets in leaf_targets.items():
            if cur_targets.count(0) > cur_targets.count(1):
                cur_node.value = 0
            else:
                cur_node.value = 1

        # Visualize the tree
        dot = visualize_tree(tree, features=feature_names)
        display(dot)

        fig, ax_steel, ax_concrete = plot(tree)

        x_min, x_max = 20, 80
        y_min, y_max = 0, 40

        # Generate the decision boundary
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )

        steel_predicts = tree.predict(
            [(cur_l, cur_a, 1) for cur_l, cur_a in zip(xx.ravel(), yy.ravel())]
        )

        steel_predicts = steel_predicts.reshape(xx.shape)
        ax_steel.contourf(
            xx,
            yy,
            steel_predicts,
            alpha=0.3,
            cmap=ListedColormap(("gray", "red", "green")),
            vmin=-1,
            vmax=1,
        )
        concrete_predicts = tree.predict(
            [(cur_l, cur_a, 0) for cur_l, cur_a in zip(xx.ravel(), yy.ravel())]
        )
        ax_concrete.contourf(
            xx,
            yy,
            concrete_predicts.reshape(xx.shape),
            alpha=0.3,
            cmap=ListedColormap(("gray", "red", "green")),
            vmin=-1,
            vmax=1,
        )

        ax_steel.scatter(
            df[df.material_type == "Steel"].load_capacity,
            df[df.material_type == "Steel"].age,
            c=df[df.material_type == "Steel"].safe,
            cmap=ListedColormap(("red", "green")),
            edgecolors="k",
            s=50,
        )
        ax_concrete.scatter(
            df[df.material_type == "Concrete"].load_capacity,
            df[df.material_type == "Concrete"].age,
            c=df[df.material_type == "Concrete"].safe,
            cmap=ListedColormap(("red", "green")),
            edgecolors="k",
            s=50,
        )
        plt.show()

    df = get_safe_unsafe_data()

    root_feature = widgets.Dropdown(
        options=["age", "load_capacity", "is_concrete"],
        description="Root Node Feature:",
        value="load_capacity",
        style={"description_width": "initial"},
    )
    root_treshold = widgets.FloatSlider(
        min=0,
        max=100,
        step=1,
        description="Root Node Value:",
        value=33,
        style={"description_width": "initial"},
    )

    left_feature = widgets.Dropdown(
        options=["age", "load_capacity", "is_concrete"],
        description="Left Node Feature:",
        value="age",
        style={"description_width": "initial"},
    )
    left_treshold = widgets.FloatSlider(
        min=0,
        max=100,
        step=1,
        description="Left Node Value:",
        value=26,
        style={"description_width": "initial"},
    )

    right_feature = widgets.Dropdown(
        options=["age", "load_capacity", "is_concrete"],
        description="Right Node Feature:",
        value="age",
        style={"description_width": "initial"},
    )
    right_treshold = widgets.FloatSlider(
        min=0,
        max=100,
        step=1,
        description="Right Node Value:",
        value=29,
        style={"description_width": "initial"},
    )

    hbox = widgets.VBox(
        [
            widgets.HBox([root_feature, root_treshold]),
            widgets.HBox([left_feature, left_treshold]),
            widgets.HBox([right_feature, right_treshold]),
        ]
    )

    # display(hbox)

    # widgets.interact(
    # widgets.interact(
    #     run,
    #     df=widgets.fixed(df),
    #     root_feature=root_feature,
    #     root_treshold=root_treshold,
    #     left_feature=left_feature,
    #     left_treshold=left_treshold,
    #     right_feature=right_feature,
    #     right_treshold=right_treshold,
    # )

    output = widgets.interactive_output(
        run,
        {
            "df": widgets.fixed(df),
            "root_feature": root_feature,
            "root_treshold": root_treshold,
            "left_feature": left_feature,
            "left_treshold": left_treshold,
            "right_feature": right_feature,
            "right_treshold": right_treshold,
        },
    )

    display(hbox, output)


def run_train_val_split_example():
    output = widgets.Output()

    # Step 3: Define a function to handle the interaction and update the output
    def run(val_size, seed):
        heart_df, feature_keys, _ = get_prepped_heart_df()

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            heart_df[feature_keys],
            heart_df["target_encoded"],
            test_size=val_size,
            random_state=seed,
        )
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)

        # Create a Decision Tree classifier
        clf = DecisionTreeClassifier(random_state=seed)

        # Train the model on the training data
        clf.fit(train_df[feature_keys], train_df["target_encoded"])

        # Get model predictions
        train_y_pred = clf.predict(train_df[feature_keys])
        val_y_pred = clf.predict(val_df[feature_keys])

        # Calculate accuracy
        train_accuracy = accuracy_score(train_df["target_encoded"], train_y_pred)
        val_accuracy = accuracy_score(val_df["target_encoded"], val_y_pred)

        with output:
            print(f"----- Validation Size: {val_size:.2f}, Seed: {seed} -----")
            print(f"Number of training samples: {len(train_df)}")
            print(f"Number of validation samples: {len(val_df)}")

            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"--------------------------------------")

    widgets.interact(
        run,
        val_size=widgets.FloatSlider(
            value=0.2,
            min=0.05,
            max=0.95,
            step=0.05,
            description="Validation size:",
            style={"description_width": "initial"},
            continuous_update=False,
        ),
        seed=widgets.IntSlider(
            min=0,
            max=100,
            step=1,
            value=42,
            continuous_update=False,
            description="Seed",
        ),
    )

    display(output)


def run_cv_example():
    heart_df, feature_keys, _ = get_prepped_heart_df()

    output = widgets.Output()

    def run(n_splits: int, seed: int):
        # Create a Decision Tree classifier
        clf = DecisionTreeClassifier(random_state=seed)

        # Use cross-validation to evaluate the model
        cur_heart_df = heart_df.sample(frac=1, random_state=seed)
        cv_scores = cross_val_score(
            clf, heart_df[feature_keys], cur_heart_df["target_encoded"], cv=n_splits
        )

        with output:
            # Print the cross-validation scores
            print(f"------ Number of splits: {n_splits}, Seed: {seed} -------")
            print(f"Number of samples per split: {len(heart_df) // n_splits}")
            print(
                f"Cross-validation scores: {', '.join([f'{cur_score:.4f}' for cur_score in cv_scores])}"
            )
            print(f"Mean cross-validation score: {cv_scores.mean():.4f}")
            print(
                f"Standard deviation of cross-validation scores: {cv_scores.std():.4f}"
            )
            print(f"---------------------------------------------------")

    widgets.interact(
        run,
        n_splits=widgets.IntSlider(
            value=5,
            min=2,
            max=10,
            step=1,
            description="Number of splits:",
            style={"description_width": "initial"},
            continuous_update=False,
        ),
        seed=widgets.IntSlider(
            min=0,
            max=100,
            step=1,
            value=42,
            continuous_update=False,
            description="Seed",
        ),
    )

    display(output)
