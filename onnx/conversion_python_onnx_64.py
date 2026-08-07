import numpy as np
import joblib
import onnx
from onnx.helper import (
    make_model,
    make_graph,
    make_node,
    make_tensor_value_info,
    make_opsetid,
    make_tensor
)
number='6'

# 1. Load the true 64-bit scikit-learn model
model_path =f"trained_lsurf_models/trained_lsurf_model_{number}.joblib"
clr = joblib.load(model_path)


# 2. Setup lists to hold our ONNX node attributes
nodes_modes = [] # The target how to go 
nodes_featureids = [] # feature id
nodes_splits = [] # threshold to do the splitting
nodes_truenodeids = [] # true branch node
nodes_trueleafs = [] # 1 for true 0 for interior node
nodes_falsenodeids = [] # false branch node
nodes_falseleafs = [] # 1 for false branch is leaf 0 if an interior node
tree_roots = [] # tree root
leaf_targetids = [] # a list of 0
leaf_weights = [] # the output

node_offset = 0
leaf_offset = 0

# 3. Parse the Scikit-Learn trees exactly as they are (in 64-bit!)
for estimator in clr.estimators_:
    tree = estimator.tree_
    tree_roots.append(node_offset)

    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value # Value

    is_leaf = children_left == -1 # np.array([1, 2, 4, -1, -1])
    # is_leaf = [False, False, False, True, True]                      
    
    onnx_node_id = {}
    onnx_leaf_id = {}
    curr_node_id = node_offset
    curr_leaf_id = leaf_offset
    
    # Map sklearn's mixed node/leaf IDs to ONNX's separated IDs
    for j in range(n_nodes):
        if is_leaf[j]:
            onnx_leaf_id[j] = curr_leaf_id
            curr_leaf_id += 1
        else:
            onnx_node_id[j] = curr_node_id
            curr_node_id += 1
            
    for j in range(n_nodes):
        if is_leaf[j]:
            leaf_targetids.append(0) 
            # CRITICAL: Keep as 64-bit float
            leaf_weights.append(np.float64(value[j][0][0])) 
        else:
            nodes_modes.append(0) # 0 = BRANCH_LEQ (<=), matching sklearn's default
            nodes_featureids.append(int(feature[j]))
            # CRITICAL: Keep as 64-bit float
            nodes_splits.append(np.float64(threshold[j])) 
            
            # Route true branch
            left_child = children_left[j]
            if is_leaf[left_child]:
                nodes_trueleafs.append(1)
                nodes_truenodeids.append(onnx_leaf_id[left_child])
            else:
                nodes_trueleafs.append(0)
                nodes_truenodeids.append(onnx_node_id[left_child])
                
            # Route false branch
            right_child = children_right[j]
            if is_leaf[right_child]:
                nodes_falseleafs.append(1)
                nodes_falsenodeids.append(onnx_leaf_id[right_child])
            else:
                nodes_falseleafs.append(0)
                nodes_falsenodeids.append(onnx_node_id[right_child])
                
    node_offset = curr_node_id
    leaf_offset = curr_leaf_id

# 4. Pack the 64-bit arrays into strictly typed ONNX Tensors
# Notice we use TensorProto.DOUBLE and np.float64 here
nodes_modes_tensor = make_tensor("nodes_modes", onnx.TensorProto.UINT8, (len(nodes_modes),), np.array(nodes_modes, dtype=np.uint8))
nodes_splits_tensor = make_tensor("nodes_splits", onnx.TensorProto.DOUBLE, (len(nodes_splits),), np.array(nodes_splits, dtype=np.float64))
leaf_weights_tensor = make_tensor("leaf_weights", onnx.TensorProto.DOUBLE, (len(leaf_weights),), np.array(leaf_weights, dtype=np.float64))

# 5. Build the newer opset 5 TreeEnsemble node
tree_node = make_node(
    "TreeEnsemble",
    inputs=[f"double_input_{number}"],
    outputs=[f"double_output_{number}"],
    domain="ai.onnx.ml",
    n_targets=1,
    aggregate_function=0, # 0 = AVERAGE (Correct for scikit-learn Random Forest)
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
    name="Manual_64Bit_Tree"
)

# 6. Wrap it in a formal ONNX Graph and Model
X_info = make_tensor_value_info(f"double_input_{number}", onnx.TensorProto.DOUBLE, [None, int(number)])
Y_info = make_tensor_value_info(f"double_output_{number}", onnx.TensorProto.DOUBLE, [None, 1])
graph = make_graph([tree_node], "rf_graph_double", [X_info], [Y_info]) # [tree_node] could be a lot of different types of nodes connecting together

# IMPORTANT: Force the model to use opset 5 for ai.onnx.ml
imp_ml = make_opsetid("ai.onnx.ml", 5) # For the extensional toolkit for TreeEnsemble
imp_onnx = make_opsetid("", 15) # For the general toolkit
onx_model64 = make_model(graph, opset_imports=[imp_onnx, imp_ml])

feature_names = [
'Mineral_source', 'uranium_total', 'Site_Density', 'U_species1','U_species8', 'U_species20']


for i,name in enumerate(feature_names):
    meta = onx_model64.metadata_props.add()
    meta.key = f"feature_{i}"
    meta.value = name

# 7. Save the final file
with open(f"lsurf_model_{number}_float_64.onnx", "wb") as f:
    f.write(onx_model64.SerializeToString())

print("True 64-bit model manually constructed and saved!!!")
