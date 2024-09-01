import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from graphviz import Digraph
import ipywidgets as widgets
from IPython.display import display
import notebook_code as nc


class DecisionLeaf:
    def __init__(
        self,
        feature_1_max: float,
        feature_1_min: float,
        feature_2_max: float,
        feature_2_min: float,
    ):
        self.class_counts = {
            "Sunny": 0,
            "Rainy": 0,
            "Cloudy": 0,
            "Snowy": 0,
        }

        self.feature_1_max = feature_1_max
        self.feature_1_min = feature_1_min
        self.feature_2_max = feature_2_max
        self.feature_2_min = feature_2_min

    def set_values(self, df: pd.DataFrame):
        for cur_class, cur_count in df.groupby("target").size().items():
            self.class_counts[cur_class] = cur_count

        self.class_counts = pd.Series(self.class_counts)

    @property
    def pred_class(self):
        if isinstance(self.class_counts, dict):
            return -1
        if self.class_counts.sum() == 0:
            return "Unknown"
        return self.class_counts.idxmax()

    @property
    def id(self):
        return str(id(self))


class DecisionTreeNode:

    def __init__(
        self,
        feature: str,
        threshold: float,
        left: "DecisionTreeNode | DecisionLeaf",
        right: "DecisionTreeNode | DecisionLeaf",
        feature_1_max: float = None,
        feature_1_min: float = None,
        feature_2_max: float = None,
        feature_2_min: float = None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

        self.feature_1_max = feature_1_max
        self.feature_1_min = feature_1_min
        self.feature_2_max = feature_2_max
        self.feature_2_min = feature_2_min

    @property
    def id(self):
        return str(id(self))

    def set_values(self, df: pd.DataFrame):
        cur_left_df = df.loc[df[self.feature] <= self.threshold]
        if isinstance(self.left, DecisionTreeNode):
            self.left.set_values(cur_left_df)
        else:
            self.left.set_values(cur_left_df)

        cur_right_df = df.loc[df[self.feature] > self.threshold]
        if isinstance(self.right, DecisionTreeNode):
            self.right.set_values(cur_right_df)
        else:
            self.right.set_values(cur_right_df)

    def get_predictions(self, df: pd.DataFrame):
        results = []

        cur_left_df = df.loc[df[self.feature] <= self.threshold].copy()
        if cur_left_df.shape[0] > 0:
            if isinstance(self.left, DecisionTreeNode):
                results.append(self.left.get_predictions(cur_left_df))
            else:
                cur_left_df.loc[:, "prediction"] = self.left.pred_class
                results.append(cur_left_df)

        cur_right_df = df.loc[df[self.feature] > self.threshold].copy()
        if cur_right_df.shape[0] > 0:
            if isinstance(self.right, DecisionTreeNode):
                results.append(self.right.get_predictions(cur_right_df))
            else:
                cur_right_df.loc[:, "prediction"] = self.right.pred_class
                results.append(cur_right_df)

        return pd.concat(results)

    @classmethod
    def build_from_dict(
        cls,
        data: dict,
        feature_1_max=999,
        feature_1_min=-999,
        feature_2_max=999,
        feature_2_min=-999,
    ):
        feature = data["feature"]
        threshold = data["threshold"]

        # Prevent non-sensical thresholds
        if feature == DecisionTree.FEATURE_1:
            threshold = min(threshold, feature_1_max)
            threshold = max(threshold, feature_1_min)
        elif feature == DecisionTree.FEATURE_2:
            threshold = min(threshold, feature_2_max)
            threshold = max(threshold, feature_2_min)

        if data["left"] is None:
            left = DecisionLeaf(
                threshold if feature == DecisionTree.FEATURE_1 else feature_1_max,
                feature_1_min,
                threshold if feature == DecisionTree.FEATURE_2 else feature_2_max,
                feature_2_min,
            )
        else:
            left = cls.build_from_dict(
                data["left"],
                threshold if feature == DecisionTree.FEATURE_1 else feature_1_max,
                feature_1_min,
                threshold if feature == DecisionTree.FEATURE_2 else feature_2_max,
                feature_2_min,
            )

        if data["right"] is None:
            right = DecisionLeaf(
                feature_1_max,
                threshold if feature == DecisionTree.FEATURE_1 else feature_1_min,
                feature_2_max,
                threshold if feature == DecisionTree.FEATURE_2 else feature_2_min,
            )
        else:
            right = cls.build_from_dict(
                data["right"],
                feature_1_max,
                threshold if feature == DecisionTree.FEATURE_1 else feature_1_min,
                feature_2_max,
                threshold if feature == DecisionTree.FEATURE_2 else feature_2_min,
            )

        return cls(
            feature,
            threshold,
            left,
            right,
            feature_1_max,
            feature_1_min,
            feature_2_max,
            feature_2_min,
        )


class DecisionTree:

    FEATURE_1 = "temperature"
    FEATURE_2 = "humidity"

    COLOR_MAPPING = {
        "Sunny": "yellow",
        "Snowy": "blue",
        "Cloudy": "red",
        "Rainy": "green",
        "Unknown": "white",
    }

    def __init__(self, root_node: DecisionTreeNode):
        self.root_node = root_node

    def set_values(self, data: pd.DataFrame):
        return self.root_node.set_values(data)

    def get_predictions(self, df: pd.DataFrame):
        return self.root_node.get_predictions(df)

    @classmethod
    def build_from_dict(cls, data: dict):
        root_node = DecisionTreeNode.build_from_dict(data)
        return cls(root_node)


def visualize_tree(tree: DecisionTree):
    dot = Digraph()

    # Add nodes recursively and create a dot object
    dot = add_nodes_edges(dot, tree.root_node)
    return dot


def add_nodes_edges(
    dot: Digraph,
    node: DecisionTreeNode | DecisionLeaf,
    parent_node: DecisionTreeNode = None,
):
    if isinstance(node, DecisionLeaf):
        dot.node(
            name=node.id,
            label=f"Leaf Value: {node.pred_class}\n"
            f"Sun: {node.class_counts['Sunny']}, R: {node.class_counts['Rainy']}, C: {node.class_counts['Cloudy']}, Snw: {node.class_counts['Snowy']}",
            # f"{node.feature_1_max:.2f}, {node.feature_1_min:.2f}\n{node.feature_2_max:.2f}, {node.feature_2_min:.2f}",
        )
    else:
        dot.node(
            name=node.id,
            label=f"{node.feature} <= {node.threshold}\n",
            # f"{node.feature_1_max:.2f}, {node.feature_1_min:.2f}\n{node.feature_2_max:.2f}, {node.feature_2_min:.2f}",
        )
        dot = add_nodes_edges(dot, node.left, node)
        dot = add_nodes_edges(dot, node.right, node)

    if parent_node is not None:
        dot.edge(parent_node.id, node.id)

    return dot


def generate_man_decision_boundary_plot(
    tree: DecisionTree, df: pd.DataFrame, show_values: bool = False
):
    fig, ax = plt.subplots(figsize=(12, 9))

    __process_node(tree.root_node, ax)

    ax.set_xlim(-30, 120)
    ax.set_ylim(10, 120)

    ax.set_xlabel(DecisionTree.FEATURE_1)
    ax.set_ylabel(DecisionTree.FEATURE_2)
    ax.set_title("Decision Boundary")

    legend_elements = [
        Patch(facecolor=color, label=class_name)
        for class_name, color in DecisionTree.COLOR_MAPPING.items()
    ]
    ax.legend(handles=legend_elements)

    if show_values:
        for cur_class, cur_df in df.groupby("target"):
            ax.scatter(
                cur_df[DecisionTree.FEATURE_1],
                cur_df[DecisionTree.FEATURE_2],
                color=DecisionTree.COLOR_MAPPING[cur_class],
                edgecolor="black",
            )

    fig.tight_layout()
    return fig, ax


def __process_node(node: DecisionTreeNode | DecisionLeaf, ax: plt.Axes):
    if isinstance(node, DecisionTreeNode):
        if node.feature == DecisionTree.FEATURE_1:
            ax.plot(
                [node.threshold, node.threshold],
                [node.feature_2_min, node.feature_2_max],
                color="black",
            )
        elif node.feature == DecisionTree.FEATURE_2:
            ax.plot(
                [node.feature_1_min, node.feature_1_max],
                [node.threshold, node.threshold],
                color="black",
            )

        __process_node(node.left, ax)
        __process_node(node.right, ax)
    else:
        ax.fill_between(
            [node.feature_1_min, node.feature_1_max],
            node.feature_2_min,
            node.feature_2_max,
            alpha=0.5,
            color=DecisionTree.COLOR_MAPPING[node.pred_class],
        )


def run_man_decision_tree():
    # Load the weather data
    features = ["temperature", "humidity"]
    df = nc.load_weather_data(fp="resources/weather_classification_data.csv",
                              subset=True, features=features, random_state=12)

    min_max_lookup = {
        "temperature": (-30, 120),
        "humidity": (10, 120)
    }

    n_levels = widgets.Dropdown(options=[1, 2, 3], value=1,
                                description="Number of levels:",
                                style={"description_width": "initial"})
    update_button = widgets.Button(description="Update")
    n_levels_box = widgets.HBox([n_levels, update_button])

    widget_output = widgets.Output()
    run_button = widgets.Button(description="Run")
    tree_output = widgets.Output()

    display(n_levels_box, widget_output, run_button, tree_output)

    global_widgets = {}

    def gen_level_widgets(tree_level: int):
        n_options = 2 ** tree_level
        cur_widgets = []
        for i in range(n_options):
            cur_feature_w = widgets.Dropdown(options=["temperature", "humidity", "leaf"],
                                             value="temperature",
                                             description=f"L{tree_level} - Split {i}")
            cur_feature_threshold_w = widgets.FloatSlider(value=0, min=-30, max=120,
                                                          description="Threshold")
            v_box = widgets.VBox([cur_feature_w, cur_feature_threshold_w])
            cur_widgets.append((cur_feature_w, cur_feature_threshold_w, v_box))
        global_widgets[tree_level] = cur_widgets

    def update_levels(b):
        widget_output.clear_output()
        with widget_output:
            # Removing levels
            n_new_levels = int(n_levels.value)
            n_existing_levels = len(global_widgets)
            # print(f"Existing levels: {n_existing_levels}, New levels: {n_new_levels}")
            if n_existing_levels > n_new_levels:
                for i in range(n_new_levels, n_existing_levels):
                    del global_widgets[i]

            # Adding new levels
            for i in range(n_existing_levels, n_new_levels):
                gen_level_widgets(i)

            for cur_level in global_widgets.keys():
                h_box = widgets.HBox(
                    [v_box for (_, _, v_box) in global_widgets[cur_level]])
                display(h_box)

    def run(b):
        tree_output.clear_output()
        with tree_output:
            tree_dict = generate_tree_dict()

            tree = DecisionTree.build_from_dict(tree_dict)
            tree.set_values(df)

            dot = visualize_tree(tree)
            display(dot)

            pred_df = tree.get_predictions(df[features]).sort_index()
            acc = (pred_df["prediction"] == df["target"]).mean()
            print(f"Accuracy: {acc:.2f}")

            fig, ax = generate_man_decision_boundary_plot(tree, df, show_values=True)
            display(fig)

    def generate_tree_dict():
        tree_dict = _gen_tree_dict(0, 0)
        return tree_dict

    def _gen_tree_dict(
            cur_level: int,
            cur_ix: int,
    ):
        if cur_level == len(global_widgets):
            return None

        cur_feature = global_widgets[cur_level][cur_ix][0].value
        if cur_feature == "leaf":
            return None

        cur_tree_dict = {
            "feature": cur_feature,
            "threshold": global_widgets[cur_level][cur_ix][1].value,
            "left": _gen_tree_dict(cur_level + 1, cur_ix * 2),
            "right": _gen_tree_dict(cur_level + 1, cur_ix * 2 + 1)
        }
        return cur_tree_dict

    update_button.on_click(update_levels)
    run_button.on_click(run)

    update_levels(None)