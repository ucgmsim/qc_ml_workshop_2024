import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib.colors import ListedColormap
import ipywidgets as widgets


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


def decision_tree():
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
    toggle1 = widgets.Checkbox(value=False, description="Toggle 1")

    toggle2 = widgets.Checkbox(value=False, description="Toggle 2")

    toggle3 = widgets.Checkbox(value=False, description="Toggle 3")
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
    heart_df, model, feature1, feature2, thal_key, features, thal_keys
):
    m = heart_df[thal_key] == 1

    x_min, x_max = heart_df[feature1].min() - 0.5, heart_df[feature1].max() + 0.5
    y_min, y_max = heart_df[feature2].min() - 0.5, heart_df[feature2].max() + 0.5

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
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(("red", "blue")))
    plt.scatter(
        heart_df.loc[m, [feature1]].values,
        heart_df.loc[m, [feature2]].values,
        c=heart_df.loc[m, "target_encoded"],
        edgecolor="k",
        cmap=ListedColormap(("red", "blue")),
    )
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f"Decision Boundary thal = {thal_key}, N = {m.sum()}")


def plot_decision_boundary_heart(heart_df, clf, features, figsize=(8, 6)):
    thal_keys = [cur_col for cur_col in heart_df.columns if cur_col.startswith("thal_")]

    # Plot decision boundary for each categorical value of 'thal'
    for cur_thal_key in thal_keys:
        fig = plt.figure(figsize=figsize)
        plot_single_decision_boundary_heart(
            heart_df, clf, "thalach", "oldpeak", cur_thal_key, features, thal_keys
        )


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
    return heart_df, features_keys


def hyperparam_tuning_example():
    heart_df, feature_keys = get_prepped_heart_df()

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(heart_df[feature_keys],
                                                      heart_df['target_encoded'],
                                                      test_size=0.2, random_state=42)

    # Create widgets for hyperparameters
    style = {'description_width': 'initial'}
    max_depth = widgets.IntSlider(value=10, min=1, max=10, step=1,
                                  description='Max Depth:', style=style)
    min_samples_split = widgets.IntSlider(value=2, min=2, max=25, step=1,
                                          description='Min Samples Split:', style=style)
    min_samples_leaf = widgets.IntSlider(value=1, min=1, max=25, step=1,
                                         description='Min Samples Leaf:', style=style)
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
                random_state=42
            )

            # Train the model on the training data
            clf.fit(X_train, y_train)

            # Get model predictions
            y_pred = clf.predict(X_val)

            # Calculate accuracy
            val_acc = accuracy_score(y_val, y_pred)
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            print(f"Training Accuracy: {train_acc:.2f}")
            print(f"Validation Accuracy: {val_acc:.2f}")

            # Visualize the decision tree
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_tree(clf, filled=True, ax=ax, feature_names=feature_keys,
                      class_names=["No Presence", "Presence"], impurity=False)
            plt.show()

    # Attach the update_plot function to the widgets
    max_depth.observe(update_plot, names='value')
    min_samples_split.observe(update_plot, names='value')
    min_samples_leaf.observe(update_plot, names='value')

    # Display the widgets and the output plot
    display(hbox, output)

    # Call the function initially to display the plot
    update_plot()