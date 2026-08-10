import json
from pathlib import Path

import joblib
import numpy as np
import onnx
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor,
    make_tensor_value_info,
)


def tree_convertor(
    model_path,
    n_features,
    x_mean,
    x_scale,
    y_mean,
    y_scale,
    input_names="double_input",
    output_names="double_output",
    shape_mode="dynamic",
    output_path=None,
):
    """Export a scaled multi-output random forest as a float64 ONNX graph."""
    # 1. Load the true 64-bit scikit-learn model
    clr = joblib.load(model_path)

    # estimators_[0].tree_.value shape is [n_nodes, n_outputs, max_n_classes])
    sample_tree = clr.estimators_[0].tree_
    n_outputs = sample_tree.value.shape[1]
    print(f"Outputs number is {n_outputs}")

    if clr.n_features_in_ != n_features:
        raise ValueError(
            f"n_features is {n_features}, but the model expects "
            f"{clr.n_features_in_}"
        )

    x_mean = np.asarray(x_mean, dtype=np.float64)
    x_scale = np.asarray(x_scale, dtype=np.float64)
    y_mean = np.asarray(y_mean, dtype=np.float64)
    y_scale = np.asarray(y_scale, dtype=np.float64)
    expected_shapes = {
        "x_mean": (x_mean, (n_features,)),
        "x_scale": (x_scale, (n_features,)),
        "y_mean": (y_mean, (n_outputs,)),
        "y_scale": (y_scale, (n_outputs,)),
    }
    for name, (values, expected_shape) in expected_shapes.items():
        print(f'name: {name}')
        print(f'value: {values}')
        print(f'value shape: {values.shape}')
        print(f'expected shape: {expected_shape}')
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {values.shape}; expected {expected_shape}"
            )

    # 2. Setup lists to hold our ONNX node attributes
    nodes_modes = []
    nodes_featureids = []
    nodes_splits = []
    nodes_truenodeids = []
    nodes_trueleafs = []
    nodes_falsenodeids = []
    nodes_falseleafs = []
    tree_roots = []
    leaf_targetids = []
    leaf_weights = []

    node_offset = 0
    leaf_offset = 0

    # 3. Parse the Scikit-Learn trees exactly as they are (in 64-bit!)
    for estimator in clr.estimators_:
        tree = estimator.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value
        is_leaf = children_left == -1   # np.array([1, 2, 4, -1, -1])
        # is_leaf = [False, False, False, True, True]      

        # TreeEnsemble v5 stores one target contribution per leaf. A
        # multi-output scikit-learn leaf stores one value for every target,
        # so duplicate each estimator topology once per target.
        for target_id in range(n_outputs):
            tree_roots.append(node_offset)
            onnx_node_id = {}
            onnx_leaf_id = {}

            for node_id in range(n_nodes):
                if is_leaf[node_id]:
                    onnx_leaf_id[node_id] = leaf_offset
                    leaf_offset += 1
                else:
                    onnx_node_id[node_id] = node_offset
                    node_offset += 1

            for node_id in range(n_nodes):
                if is_leaf[node_id]:
                    leaf_targetids.append(target_id)
                    leaf_weights.append(
                        np.float64(value[node_id][target_id][0])
                    )
                    continue

                nodes_modes.append(0)
                nodes_featureids.append(int(feature[node_id]))
                nodes_splits.append(np.float64(threshold[node_id]))

                left_child = children_left[node_id]
                left_is_leaf = bool(is_leaf[left_child])
                nodes_trueleafs.append(int(left_is_leaf))
                nodes_truenodeids.append(
                    onnx_leaf_id[left_child]
                    if left_is_leaf
                    else onnx_node_id[left_child]
                )

                right_child = children_right[node_id]
                right_is_leaf = bool(is_leaf[right_child])
                nodes_falseleafs.append(int(right_is_leaf))
                nodes_falsenodeids.append(
                    onnx_leaf_id[right_child]
                    if right_is_leaf
                    else onnx_node_id[right_child]
                )

    # 4. Pack the 64-bit arrays into strictly typed ONNX Tensors
    # Notice we use TensorProto.DOUBLE and np.float64 here
    nodes_modes_tensor = make_tensor(
        "nodes_modes",
        onnx.TensorProto.UINT8,
        (len(nodes_modes),),
        np.array(nodes_modes, dtype=np.uint8),
    )
    nodes_splits_tensor = make_tensor(
        "nodes_splits",
        onnx.TensorProto.DOUBLE,
        (len(nodes_splits),),
        np.array(nodes_splits, dtype=np.float64),
    )
    leaf_weights_tensor = make_tensor(
        "leaf_weights",
        onnx.TensorProto.DOUBLE,
        (len(leaf_weights),),
        np.array(leaf_weights, dtype=np.float64),
    )

    x_mean_init = make_tensor(
        "x_mean", onnx.TensorProto.DOUBLE, [n_features], x_mean
    )
    x_scale_init = make_tensor(
        "x_scale", onnx.TensorProto.DOUBLE, [n_features], x_scale
    )
    y_mean_init = make_tensor(
        "y_mean", onnx.TensorProto.DOUBLE, [n_outputs], y_mean
    )
    y_scale_init = make_tensor(
        "y_scale", onnx.TensorProto.DOUBLE, [n_outputs], y_scale
    )
    estimator_count_init = make_tensor(
        "estimator_count",
        onnx.TensorProto.DOUBLE,
        [],
        [np.float64(len(clr.estimators_))],
    )

    sub_node = make_node(
        "Sub", inputs=[input_names, "x_mean"], outputs=["x_sub"], name="Pre_Sub"
    )
    div_node = make_node(
        "Div", inputs=["x_sub", "x_scale"], outputs=["x_scaled"], name="Pre_Div"
    )
    # RandomForestRegressor converts prediction inputs to float32 before tree
    # traversal. Preserve double precision in the scaler graph, then reproduce
    # that routing behavior before comparing against the stored thresholds.
    cast_to_float = make_node(
        "Cast",
        inputs=["x_scaled"],
        outputs=["x_scaled_float"],
        to=onnx.TensorProto.FLOAT,
        name="Tree_Input_To_Float32",
    )
    cast_to_double = make_node(
        "Cast",
        inputs=["x_scaled_float"],
        outputs=["tree_input"],
        to=onnx.TensorProto.DOUBLE,
        name="Tree_Input_To_Float64",
    )
    
    # 5. Build the newer opset 5 TreeEnsemble node
    tree_node = make_node(
        "TreeEnsemble",
        inputs=["tree_input"],
        outputs=["y_sum"],
        domain="ai.onnx.ml",
        n_targets=n_outputs,
        # SUM followed by division by the number of estimators is required for
        # multi-output forests because each target has a duplicated topology.
        aggregate_function=1,   # For multiple output tree
        tree_roots=tree_roots,
        nodes_modes=nodes_modes_tensor,
        nodes_featureids=nodes_featureids,
        nodes_splits=nodes_splits_tensor,
        nodes_truenodeids=nodes_truenodeids,
        nodes_trueleafs=nodes_trueleafs,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_falseleafs=nodes_falseleafs,
        leaf_targetids=leaf_targetids,
        leaf_weights=leaf_weights_tensor,
        name="TreeEnsemble",
    )
    average_node = make_node(
        "Div",
        inputs=["y_sum", "estimator_count"],
        outputs=["y_scaled"],
        name="Forest_Average",
    )

    if shape_mode == "dynamic":
        in_shape, out_shape = ["dynamic", n_features], ["dynamic", n_outputs]
    elif shape_mode == "single":
        in_shape, out_shape = [1, n_features], [1, n_outputs]
    elif shape_mode == "feature_only":
        in_shape, out_shape = [n_features], [n_outputs]
    elif shape_mode == "scalar":
        in_shape, out_shape = [], []
    else:
        raise ValueError(f"Unknown shape_mode: {shape_mode}")

    mul_node = make_node(
        "Mul",
        inputs=["y_scaled", "y_scale"],
        outputs=["y_mul"],
        name="Post_Mul",
    )
    add_node = make_node(
        "Add",
        inputs=["y_mul", "y_mean"],
        outputs=[output_names],
        name="Post_Add",
    )

    # 6. Wrap it in a formal ONNX Graph and Model
    X_info = make_tensor_value_info(input_names, onnx.TensorProto.DOUBLE, in_shape)
    Y_info = make_tensor_value_info(output_names, onnx.TensorProto.DOUBLE, out_shape)
    graph = make_graph(
        [
            sub_node,
            div_node,
            cast_to_float,
            cast_to_double,
            tree_node,
            average_node,
            mul_node,
            add_node,
        ],
        "rf_graph_double",
        [X_info],
        [Y_info],
        initializer=[
            x_mean_init,
            x_scale_init,
            y_mean_init,
            y_scale_init,
            estimator_count_init,
        ],
    )

    # IMPORTANT: Force the model to use opset 5 for ai.onnx.ml
    imp_ml = make_opsetid("ai.onnx.ml", 5)
    imp_onnx = make_opsetid("", 15)
    onx_model64 = make_model(graph, opset_imports=[imp_onnx, imp_ml], ir_version=11)

    onnx.checker.check_model(onx_model64, full_check=True)

    # 7. Save the final file
    if output_path is None:
        output_path = Path("models/alsurf_rf") / f"alsurf_{n_features}.onnx"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onx_model64, output_path)

    print(f"True 64-bit model manually constructed and saved to {output_path}")
    return output_path


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]   # Resolve the repository root directory (two levels up).
    model_directory = repository_root / "models" / "alsurf_rf"
    n_feature = 6
    rf_model = model_directory / f"alsurf_{n_feature}.joblib"
    x_scalar = joblib.load(model_directory / "x_scalar.joblib")
    y_scalar = joblib.load(model_directory / "y_scalar.joblib")

    original_features = list(x_scalar.feature_names_in_)
    print(f"Original features types: {type(original_features)}")
    print(f"Original features: {original_features}")

    with (model_directory / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
        selected_features = metadata["selected_features"][n_feature - 1]

    print(f"Selected features are: {selected_features}")

    selected_indices = [original_features.index(f) for f in selected_features]
    print(f"Selected indices: {selected_indices}")
    x_mean = np.array(x_scalar.mean_[selected_indices], dtype=np.float64)
    x_scale = np.array(x_scalar.scale_[selected_indices], dtype=np.float64)

    y_mean = np.array(y_scalar.mean_, dtype=np.float64)
    y_scale = np.array(y_scalar.scale_, dtype=np.float64)

    tree_convertor(
        rf_model,
        n_feature,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
        output_path=model_directory / f"alsurf_{n_feature}.onnx",
    )
