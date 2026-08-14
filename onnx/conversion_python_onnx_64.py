import argparse
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
    feature_names=None,
    target_names=None,
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
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {values.shape}; expected {expected_shape}"
            )

    shape_aliases = {
        "dynamic": "dynamic_batch",
        "dynamic_batch": "dynamic_batch",
        "single": "batch1",
        "batch1": "batch1",
        "feature_only": "feature_vector",
        "feature_vector": "feature_vector",
        "scalar": "scalar",
    }
    try:
        shape_mode = shape_aliases[shape_mode]
    except KeyError as error:
        valid_modes = ", ".join(sorted(shape_aliases))
        raise ValueError(
            f"Unknown shape_mode {shape_mode!r}; expected one of {valid_modes}"
        ) from error

    if feature_names is None:
        feature_names = [f"feature_{index}" for index in range(n_features)]
    if target_names is None:
        target_names = [f"target_{index}" for index in range(n_outputs)]
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries; expected {n_features}"
        )
    if len(target_names) != n_outputs:
        raise ValueError(
            f"target_names has {len(target_names)} entries; expected {n_outputs}"
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

    input_wrapper_nodes = []
    output_wrapper_nodes = []
    signature_initializers = []

    if shape_mode in {"dynamic_batch", "batch1"}:
        if not isinstance(input_names, str) or not isinstance(output_names, str):
            raise TypeError(
                "Batch models require one string input name and one string "
                "output name"
            )
        batch_dimension = "batch_size" if shape_mode == "dynamic_batch" else 1
        graph_inputs = [
            make_tensor_value_info(
                input_names,
                onnx.TensorProto.DOUBLE,
                [batch_dimension, n_features],
            )
        ]
        graph_outputs = [
            make_tensor_value_info(
                output_names,
                onnx.TensorProto.DOUBLE,
                [batch_dimension, n_outputs],
            )
        ]
        scaler_input = input_names
        postprocess_output = output_names
    elif shape_mode == "feature_vector":
        # Expand dimension for the scalar
        if not isinstance(input_names, str) or not isinstance(output_names, str):
            raise TypeError(
                "Feature-vector models require one string input name and one "
                "string output name"
            )
        batch_axis = make_tensor(
            "batch_axis", onnx.TensorProto.INT64, [1], np.array([0], dtype=np.int64)
        )
        signature_initializers.append(batch_axis)
        input_wrapper_nodes.append(
            make_node(
                "Unsqueeze",
                inputs=[input_names, "batch_axis"],
                outputs=["model_input_2d"],
                name="Feature_Vector_To_Batch",
            )
        )
        output_wrapper_nodes.append(
            make_node(
                "Squeeze",
                inputs=["model_output_2d", "batch_axis"],
                outputs=[output_names],
                name="Batch_To_Feature_Vector",
            )
        )
        graph_inputs = [
            make_tensor_value_info(
                input_names, onnx.TensorProto.DOUBLE, [n_features]
            )
        ]
        graph_outputs = [
            make_tensor_value_info(
                output_names, onnx.TensorProto.DOUBLE, [n_outputs]
            )
        ]
        scaler_input = "model_input_2d"
        postprocess_output = "model_output_2d"
    else:
        if isinstance(input_names, str):
            input_names = [
                f"{input_names}_{index}" for index in range(n_features)
            ]
        if isinstance(output_names, str):
            output_names = [
                f"{output_names}_{index}" for index in range(n_outputs)
            ]
        if len(input_names) != n_features:
            raise ValueError(
                f"Scalar mode has {len(input_names)} inputs; expected {n_features}"
            )
        if len(output_names) != n_outputs:
            raise ValueError(
                f"Scalar mode has {len(output_names)} outputs; expected {n_outputs}"
            )
        # Increase the dimension for the input
        scalar_axes = make_tensor(
            "scalar_axes",
            onnx.TensorProto.INT64,
            [2],
            np.array([0, 1], dtype=np.int64),
        )
        # Decrease the dimension for the output
        batch_axis = make_tensor(
            "batch_axis", onnx.TensorProto.INT64, [1], np.array([0], dtype=np.int64)
        )
        signature_initializers.extend([scalar_axes, batch_axis])

        packed_inputs = []
        graph_inputs = []
        for index, input_name in enumerate(input_names):
            packed_name = f"scalar_input_{index}_2d"
            packed_inputs.append(packed_name)
            graph_inputs.append(
                make_tensor_value_info(input_name, onnx.TensorProto.DOUBLE, [])
            )
            input_wrapper_nodes.append(
                make_node(
                    "Unsqueeze",
                    inputs=[input_name, "scalar_axes"],
                    outputs=[packed_name],
                    name=f"Pack_Scalar_Input_{index}",
                )
            )
        input_wrapper_nodes.append(
            make_node(
                "Concat",
                inputs=packed_inputs,
                outputs=["model_input_2d"],
                axis=1,
                name="Pack_Scalar_Inputs",
            )
        )

        output_wrapper_nodes.append(
            make_node(
                "Squeeze",
                inputs=["model_output_2d", "batch_axis"],
                outputs=["model_output_1d"],
                name="Remove_Output_Batch",
            )
        )
        graph_outputs = []
        for index, output_name in enumerate(output_names):
            index_name = f"output_index_{index}"
            signature_initializers.append(
                make_tensor(
                    index_name,
                    onnx.TensorProto.INT64,
                    [],
                    [np.int64(index)],
                )
            )
            output_wrapper_nodes.append(
                make_node(
                    "Gather",
                    inputs=["model_output_1d", index_name],
                    outputs=[output_name],
                    axis=0,
                    name=f"Unpack_Scalar_Output_{index}",
                )
            )
            graph_outputs.append(
                make_tensor_value_info(output_name, onnx.TensorProto.DOUBLE, [])
            )
        scaler_input = "model_input_2d"
        postprocess_output = "model_output_2d"

    sub_node = make_node(
        "Sub", inputs=[scaler_input, "x_mean"], outputs=["x_sub"], name="Pre_Sub"
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

    mul_node = make_node(
        "Mul",
        inputs=["y_scaled", "y_scale"],
        outputs=["y_mul"],
        name="Post_Mul",
    )
    add_node = make_node(
        "Add",
        inputs=["y_mul", "y_mean"],
        outputs=[postprocess_output],
        name="Post_Add",
    )

    # 6. Wrap it in a formal ONNX Graph and Model
    graph = make_graph(
        [
            *input_wrapper_nodes,
            sub_node,
            div_node,
            cast_to_float,
            cast_to_double,
            tree_node,
            average_node,
            mul_node,
            add_node,
            *output_wrapper_nodes,
        ],
        "rf_graph_double",
        graph_inputs,
        graph_outputs,
        initializer=[
            x_mean_init,
            x_scale_init,
            y_mean_init,
            y_scale_init,
            estimator_count_init,
            *signature_initializers,
        ],
    )

    # IMPORTANT: Force the model to use opset 5 for ai.onnx.ml
    imp_ml = make_opsetid("ai.onnx.ml", 5)
    imp_onnx = make_opsetid("", 15)
    onx_model64 = make_model(graph, opset_imports=[imp_onnx, imp_ml], ir_version=11)

    metadata_values = {"shape_mode": shape_mode}
    metadata_values.update(
        {
            f"feature_{index}": str(feature_name)
            for index, feature_name in enumerate(feature_names)
        }
    )
    metadata_values.update(
        {
            f"target_{index}": str(target_name)
            for index, target_name in enumerate(target_names)
        }
    )
    for key, value in metadata_values.items():
        metadata_property = onx_model64.metadata_props.add()
        metadata_property.key = key
        metadata_property.value = value

    onnx.checker.check_model(onx_model64, full_check=True)

    # 7. Save the final file
    if output_path is None:
        output_path = (
            Path("models/alsurf_rf")
            / f"alsurf_{n_features}_{shape_mode}.onnx"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onx_model64, output_path)

    print(f"True 64-bit model manually constructed and saved to {output_path}")
    return output_path


def export_alsurf_variants(
    model_path,
    x_scaler_path,
    y_scaler_path,
    metadata_path,
    output_directory,
    n_features=9,
):
    """Export all tensor signatures used by the Alquimia interface tests."""
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    with Path(metadata_path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)

    # Initiate the scalar with pandas
    # Find the selected features
    all_features = list(x_scaler.feature_names_in_)
    selected_features = metadata["selected_features"][n_features - 1]
    selected_indices = [
        all_features.index(feature_name) for feature_name in selected_features
    ]
    x_mean = np.asarray(x_scaler.mean_[selected_indices], dtype=np.float64)
    x_scale = np.asarray(x_scaler.scale_[selected_indices], dtype=np.float64)
    y_mean = np.asarray(y_scaler.mean_, dtype=np.float64)
    y_scale = np.asarray(y_scaler.scale_, dtype=np.float64)
    target_names = ["SURF-H+", "SURF-Zn++"]

    output_directory = Path(output_directory)
    variants = {
        "dynamic_batch": {
            "input_names": "chemical_input_raw",
            "output_names": "sorbed_output_raw",
        },
        "batch1": {
            "input_names": "chemical_input_raw",
            "output_names": "sorbed_output_raw",
        },
        "feature_vector": {
            "input_names": "chemical_input_raw",
            "output_names": "sorbed_output_raw",
        },
        "scalar": {
            "input_names": selected_features,
            "output_names": target_names,
        },
    }

    output_paths = {}
    for shape_mode, tensor_names in variants.items():
        output_path = output_directory / f"alsurf_{n_features}_{shape_mode}.onnx"
        output_paths[shape_mode] = tree_convertor(
            model_path=model_path,
            n_features=n_features,
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
            input_names=tensor_names["input_names"],
            output_names=tensor_names["output_names"],
            shape_mode=shape_mode,
            output_path=output_path,
            feature_names=selected_features,
            target_names=target_names,
        )
    return output_paths


def main():
    repository_root = Path(__file__).resolve().parents[1]
    default_model_directory = repository_root / "models" / "alsurf_rf"
    parser = argparse.ArgumentParser(
        description="Export ALSURF random-forest ONNX tensor-shape variants."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model_directory / "alsurf_9.joblib",
    )
    parser.add_argument(
        "--x-scaler-path",
        type=Path,
        default=default_model_directory / "x_scalar.joblib",
    )
    parser.add_argument(
        "--y-scaler-path",
        type=Path,
        default=default_model_directory / "y_scalar.joblib",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=default_model_directory / "metadata.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=default_model_directory,
    )
    parser.add_argument("--feature-count", type=int, default=9)
    arguments = parser.parse_args()

    export_alsurf_variants(
        model_path=arguments.model_path,
        x_scaler_path=arguments.x_scaler_path,
        y_scaler_path=arguments.y_scaler_path,
        metadata_path=arguments.metadata_path,
        output_directory=arguments.output_directory,
        n_features=arguments.feature_count,
    )


if __name__ == "__main__":
    main()
