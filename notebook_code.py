import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import ipywidgets as widgets
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
    PolynomialFeatures,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from IPython.display import display


### Data Loading
def load_weather_data(
    fp: str = "resources/weather_classification_data.csv", with_missing_values=False, features=None, subset=False, pre_process=False, random_state=42,
):
    # Load data
    df = pd.read_csv(fp)

    # Rename columns
    col_rename_dict = {
        "Temperature": "temperature",
        "Humidity": "humidity",
        "Wind Speed": "wind_speed",
        "Precipitation (%)": "precipitation_chance",
        "Cloud Cover": "cloud_cover",
        "Atmospheric Pressure": "pressure",
        "UV Index": "uv_index",
        "Season": "season",
        "Visibility (km)": "visibility",
        "Location": "location_type",
        "Weather Type": "target",
    }
    df = df.rename(columns=col_rename_dict)

    # Feature column names
    avail_features = list(col_rename_dict.values())[:-1]
    avail_cat_features = ["cloud_cover", "season", "location_type"]
    avail_num_features = list(set(avail_features) - set(avail_cat_features))

    if features is not None:
        df = df[features + ["target"]]

    if subset:
        results = []
        target_types = df["target"].unique()
        for cur_target in target_types:
            cur_df = df.loc[df["target"] == cur_target].sample(
                frac=np.random.uniform(0.09, 0.11), random_state=random_state
            )
            results.append(cur_df)

        df = pd.concat(results).sample(frac=1).reset_index(drop=True)

    # Add some fake missing values
    n_rand_values = 100
    if with_missing_values:
        rand_rows = np.random.choice(df.shape[0], n_rand_values)
        rand_cols = np.random.choice(df.shape[1] - 1, n_rand_values)

        for i in range(n_rand_values):
            df.iat[rand_rows[i], rand_cols[i]] = np.nan

    if pre_process:
        return run_weather_preprocess(df)

    return df


def load_weather_data_pre(with_missing_values=False):
    df = load_weather_data(
        with_missing_values=with_missing_values,
        features=["temperature", "humidity", "location_type"],
        subset=True,
    )

    return df


def run_weather_preprocess(df, val_df=None):
    # Standardise numerical features
    numerical_features = [
        "temperature",
        "humidity",
        "wind_speed",
        "precipitation_chance",
        "pressure",
        "uv_index",
        "visibility",
    ]
    numerical_features = list(set(numerical_features) & set(df.columns))
    scaler = StandardScaler()
    scaler.fit(df[numerical_features])
    df[numerical_features] = scaler.transform(df[numerical_features])

    # Encode categorial features
    nominal_features = ["cloud_cover", "season", "location_type"]
    nominal_features = list(set(nominal_features) & set(df.columns))
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(df[nominal_features])
    encoded_columns = encoder.get_feature_names_out(nominal_features)
    df[encoded_columns] = encoder.transform(df[nominal_features])

    # Encode target variable
    label_encoder = LabelEncoder()
    df["target_encoded"] = label_encoder.fit_transform(df["target"])

    in_features = numerical_features + list(encoded_columns)

    if val_df is not None:
        val_df[numerical_features] = scaler.transform(val_df[numerical_features])
        val_df[encoded_columns] = encoder.transform(val_df[nominal_features])
        val_df["target_encoded"] = label_encoder.transform(val_df["target"])
        return df, val_df, in_features, label_encoder

    return df, in_features, label_encoder


### Feature Scaling Example
def vis_scaling(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(df.x1, df.y, c="r", label="X1")
    ax1.scatter(df.x2, df.y, c="g", label="X2")
    ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_xlabel("Features: X1, X2")
    ax1.set_ylabel("Target")
    ax1.set_title("Original Data")
    ax1.legend()

    ax2.scatter(df.x1_std, df.y, c="r", label="X1")
    ax2.scatter(df.x2_std, df.y, c="g", label="X2")
    ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_xlabel("Standardised Features: X1, X2")
    ax2.set_title("Standardised Data")
    ax2.set_ylabel("Target")
    fig.tight_layout()


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


### Categorial
def load_cat_example():
    df = pd.DataFrame(
        [
            ["green", "M", 10.1, "class1"],
            ["red", "L", 13.5, "class2"],
            ["blue", "XL", 15.3, "class1"],
        ]
    )
    df.columns = ["color", "size", "price", "target"]

    return df


### Decision Boundary
def plot_single_decision_boundary(
    train_df,
    model,
    feature_1,
    feature_2,
    cat_feature,
    cat_features,
    ax,
    label_mapping,
    val_df=None,
    legend=False,
    y_label=True,
    show_train=True,
):
    # Get mask for current category
    train_mask = train_df[cat_feature] == 1
    cur_train_df = train_df[train_mask]

    if val_df is not None:
        val_mask = val_df[cat_feature] == 1
        cur_val_df = val_df[val_mask]

    # Determine min/max values
    x_min = train_df[feature_1].min()
    x_max = train_df[feature_1].max()
    y_min = train_df[feature_2].min()
    y_max = train_df[feature_2].max()
    if val_df is not None:
        x_min = min(x_min, val_df[feature_1].min())
        x_max = max(x_max, val_df[feature_1].max())
        y_min = min(y_min, val_df[feature_2].min())
        y_max = max(y_max, val_df[feature_2].max())

    # Generate the grid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000)
    )

    X = pd.DataFrame({feature_1: xx.ravel(), feature_2: yy.ravel()})
    X[cat_features] = 0
    X[cat_feature] = 1

    Z = model.predict(X[model.feature_names_in_])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    colors = ["red", "green", "blue", "yellow"]
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(colors), vmin=0, vmax=3)
    for cur_class in range(4):
        if show_train:
            cur_train_mask = cur_train_df.target_encoded == cur_class
            ax.scatter(
                cur_train_df.loc[cur_train_mask, feature_1],
                cur_train_df.loc[cur_train_mask, feature_2],
                c=colors[cur_class],
                label=f"Class {label_mapping[cur_class]}",
                marker="o",
                s=15,
                edgecolors="k",
            )
        if val_df is not None:
            cur_val_mask = cur_val_df.target_encoded == cur_class
            ax.scatter(
                cur_val_df.loc[cur_val_mask, feature_1],
                cur_val_df.loc[cur_val_mask, feature_2],
                c=colors[cur_class],
                label=f"Class {label_mapping[cur_class]}" if not show_train else None,
                marker="^",
                s=50,
                edgecolors="k",
            )
    if legend:
        ax.legend()

    ax.grid(False)
    ax.set_xlabel(feature_1)
    if y_label:
        ax.set_ylabel(feature_2)
    ax.set_title(f"Decision Boundary {cat_feature}, N (Training) = {train_mask.sum()}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())


def plot_decision_boundaries(
    train_df, clf, features, label_encoder, val_df=None, show_train=True
):
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

    fig, axs = plt.subplots(1, 3, figsize=(3 * 8, 8))
    cat_features = [
        cur_feature for cur_feature in features if cur_feature.startswith("location")
    ]
    for ix, (cur_cat_feature, cur_ax) in enumerate(zip(cat_features, axs.ravel())):
        plot_single_decision_boundary(
            train_df,
            clf,
            "temperature",
            "humidity",
            cur_cat_feature,
            cat_features,
            cur_ax,
            label_mapping,
            legend=ix == 0,
            y_label=ix == 0,
            val_df=val_df,
            show_train=show_train,
        )
        # if ix > 0:
        #     cur_ax.set_yticklabels([])
    fig.tight_layout()


### Overfitting Example
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


### Hyperparameter Tuning
def manual_hyperparam_tuning():
    features = ["temperature", "humidity", "location_type"]
    df = load_weather_data(with_missing_values=False, features=features, subset=True)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        df[features], df["target"], test_size=0.2, random_state=42
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)

    # Run pre-processsing
    train_df, val_df, feature_keys, label_encoder = run_weather_preprocess(train_df, val_df)

    # Create widgets for hyperparameters
    style = {"description_width": "initial"}
    max_depth = widgets.IntSlider(
        value=10,
        min=1,
        max=8,
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
                class_names=label_encoder.classes_,
                impurity=False,
            )

            plot_decision_boundaries(train_df, clf, feature_keys, label_encoder, val_df=val_df, show_train=False)
            plt.show()

    # Attach the update_plot function to the widgets
    max_depth.observe(update_plot, names="value")
    min_samples_split.observe(update_plot, names="value")
    min_samples_leaf.observe(update_plot, names="value")

    # Display the widgets and the output plot
    display(hbox, output)

    # Call the function initially to display the plot
    update_plot()

### Holdout variations plot
def holdout_variations():
    # Data loading
    features = ["temperature", "humidity", "location_type"]
    df = load_weather_data(subset=True, features=features)

    n_runs = 25
    val_acc, train_acc = [], []
    for ix in range(n_runs):
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(df[features],
                                                          df["target"],
                                                          test_size=0.2)
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)

        # Run the pre-processing
        train_df, val_df, in_features, label_encoder = run_weather_preprocess(train_df,
                                                                                 val_df)

        # Create a Decision Tree classifier
        clf = DecisionTreeClassifier(max_depth=4)

        # Train the model on the training data
        clf.fit(train_df[in_features], train_df["target_encoded"])

        # Get model predictions
        train_y_pred = clf.predict(train_df[in_features])
        val_y_pred = clf.predict(val_df[in_features])

        # Calculate accuracy
        cur_train_accuracy = accuracy_score(train_df["target_encoded"], train_y_pred)
        cur_val_accuracy = accuracy_score(val_df["target_encoded"], val_y_pred)

        # Append to the list
        train_acc.append(cur_train_accuracy)
        val_acc.append(cur_val_accuracy)

    fig = plt.figure(figsize=(8, 6))

    plt.scatter(range(n_runs), train_acc, label="Train Accuracy")
    plt.scatter(range(n_runs), val_acc, label="Validation Accuracy")
    plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    plt.tight_layout()
    plt.ylim(0.5, 1.0)
    plt.legend()

    plt.xlabel("Run")
    plt.ylabel("Accuracy")


### Interactive holdout
def run_interactive_train_val_split():
    output = widgets.Output()

    # Step 3: Define a function to handle the interaction and update the output
    def run(val_size, seed):
        features = ["temperature", "humidity", "location_type"]
        df = load_weather_data(with_missing_values=False, features=features, subset=True)

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            df[features], df["target"], test_size=0.2, random_state=42
        )
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)

        # Run pre-processsing
        train_df, val_df, feature_keys, label_encoder = run_weather_preprocess(train_df, val_df)

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


### Heart disease dataset
def load_heart_df(features: list[str] = None, drop_na: bool = True):
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
    df["target"] = df["target"].apply(lambda x: "Presence" if x == 0 else "No Presence")

    return df


### Manual Decision Tree
