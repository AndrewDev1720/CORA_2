import os
import numpy as np
import torch
from torch_geometric.datasets import FacebookPagePage
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
import sys
from utils.argument import arg_parse_exp_node_facebookpp  # Updated for Facebook Pages
from time import time

# Import your utilities
from utils.utils import calculate_Metrics, sample_according_to_l_hop_density

# Import your explainer model
from models.explainer_models import NodeExplainerEdgeMulti
from models.gcn import get_model


if __name__ == "__main__":
    # Increase print options for large adjacency matrices
    np.set_printoptions(threshold=sys.maxsize)

    # Parse experiment arguments for Facebook Pages
    exp_args = arg_parse_exp_node_facebookpp()
    print("Arguments:\n", exp_args)

    # Load the Facebook Pages dataset
    dataset_path = 'datasets/facebookpagepage/facebookpagepage_processed.pt'  # Ensure this path is correct
    facebookpp_dir = 'datasets/facebookpagepage'
    os.makedirs(facebookpp_dir, exist_ok=True)
    data = FacebookPagePage(root=facebookpp_dir)[0]
    num_nodes = data.num_nodes  # num_nodes
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[:int(0.8 * num_nodes)] = True  # 80% for training

    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask[int(0.8 * num_nodes):int(0.9 * num_nodes)] = True  # 10% for validation

    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[int(0.9 * num_nodes):] = True  # 10% for testing

    save_path = os.path.join(facebookpp_dir, "facebookpagepage_processed.pt")
    torch.save(data, save_path)  # save data

    # Set the device
    device = torch.device(f'cuda:{exp_args.cuda}') if exp_args.gpu else torch.device('cpu')
    print(f"Using device: {device}")
    data = data.to(device)

    # Load the pretrained model
    model_path = exp_args.model_path
    pretrained_model_filename = 'facebookpagepage_gcn_model.pth'
    pretrained_model_path = os.path.join(model_path, pretrained_model_filename)

    # Initialize the model
    input_dim = data.num_node_features
    hidden_dim = 16  # Same as Cora and PubMed; adjust if needed
    output_dim = 4

    model = get_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)

    # Load the pre-trained model if it exists
    if os.path.exists(pretrained_model_path):
        try:
            model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            print(f"Loaded pre-trained model from {pretrained_model_path}")
        except Exception as e:
            print(f"Error loading model state: {e}")
            sys.exit(1)
    else:
        print(f"Pre-trained model not found at {pretrained_model_path}. Please ensure the model is trained and saved correctly.")
        sys.exit(1)

    model.eval()

    # Create the explainer
    explainer = NodeExplainerEdgeMulti(
        base_model=model,
        G_dataset=data,
        args=exp_args,
        test_indices=data.test_mask.nonzero(as_tuple=True)[0]
    )

    # Sample nodes according to l-hop density
    listNode, average = sample_according_to_l_hop_density(data, 3, 0.01,42)
    print(f"Sampled {len(listNode)} nodes with l-hop density of {average:.4f}")

    # Run the explainer
    start = time()
    exp_dict, num_dict = explainer.explain_nodes_gnn_stats(listNode)
    end = time()
    duration = end - start
    print(f'Duration: {duration:.2f} seconds')

    # Extract original graph details
    x = data.x
    y = data.y
    original_edge_index = data.edge_index
    original_adj = to_dense_adj(data.edge_index)[0]

    # Initialize dictionaries to store subgraphs
    exp_sub_graph = {}
    comp_sub_graph = {}

    # Iterate over each node's explanation
    for node_id, masked_adj in exp_dict.items():
        # Ensure masked_adj is on CPU
        masked_adj = masked_adj.cpu()
        print(f"\nMasked adjacency matrix for node {node_id}:")
        print(masked_adj)

        # Count the number of edges above the threshold
        num_nonzero = (masked_adj > exp_args.mask_thresh).sum().item()
        print(f"Node {node_id}: masked_adj has {num_nonzero} edges above the threshold.")

        # Extract edges where mask is above the threshold
        rows, cols = np.where(masked_adj == 1)
        num_edges_exp = len(rows)
        print(num_edges_exp)
        exp_edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        print(f"Number of explanation edges for node {node_id}: {num_edges_exp}")

        # Create the explanation subgraph Data object
        exp_subgraph = Data(
            x=x.clone(),
            edge_index=exp_edge_index,
            y=y.clone()
        )
        exp_sub_graph[node_id] = exp_subgraph

        # Compute complement edges (edges not in the explanation)
        original_edges_set = set(map(tuple, original_edge_index.cpu().numpy().T.tolist()))
        exp_edges_set = set(map(tuple, exp_edge_index.cpu().numpy().T.tolist()))
        complement_edges_set = original_edges_set - exp_edges_set

        if len(complement_edges_set) > 0:
            comp_edge_index = torch.tensor(list(complement_edges_set), dtype=torch.long).T
        else:
            comp_edge_index = torch.empty((2, 0), dtype=torch.long)  

        # Create the complement subgraph Data object
        comp_subgraph = Data(
            x=x.clone(),
            edge_index=comp_edge_index,
            y=y.clone()
        )
        comp_sub_graph[node_id] = comp_subgraph

        # Optional: Print the number of edges in both subgraphs
        print(f"Node {node_id}:")
        print(f"  Explanation edges: {exp_edge_index.size(1)}")
        print(f"  Complement edges: {comp_edge_index.size(1)}")

    metrics_list = []

    # Iterate over each node to calculate metrics
    for node_id in exp_sub_graph.keys():
        exp_subgraph = exp_sub_graph[node_id]
        comp_subgraph = comp_sub_graph[node_id]

        # Ensure node_id is an integer
        if isinstance(node_id, torch.Tensor):
            node_idx = node_id.item()
        else:
            node_idx = node_id

        # Move subgraphs to the same device as the model
        device = next(model.parameters()).device
        exp_subgraph = exp_subgraph.to(device)
        comp_subgraph = comp_subgraph.to(device)

        # Calculate metrics
        fidelity_Plus, fidelity_Minus, sparsity_edge, sparsity_size = calculate_Metrics(
            original_graph=data,
            exp_subgraph=exp_subgraph,
            comp_subgraph=comp_subgraph,
            model=model,
            node_idx=node_idx
        )

        # Store the metrics
        metrics = {
            'node_id': node_idx,
            'fidelity_Plus': fidelity_Plus,
            'fidelity_Minus': fidelity_Minus,
            'sparsity_edge': sparsity_edge,
            'sparsity_size': sparsity_size
        }
        metrics_list.append(metrics)

        # Print the metrics for this node
        print(f"\nNode {node_idx}:")
        print(f"  Fidelity Plus: {fidelity_Plus:.4f}")
        print(f"  Fidelity Minus: {fidelity_Minus:.4f}")
        print(f"  Sparsity Edge: {sparsity_edge:.4f}")
        print(f"  Sparsity Size: {sparsity_size:.4f}")

    # Compute and print average metrics
    if metrics_list:
        avg_fidelity_Plus = np.mean([m['fidelity_Plus'] for m in metrics_list])
        avg_fidelity_Minus = np.mean([m['fidelity_Minus'] for m in metrics_list])
        avg_sparsity_edge = np.mean([m['sparsity_edge'] for m in metrics_list])
        avg_sparsity_size = np.mean([m['sparsity_size'] for m in metrics_list])

        print("\nAverage Metrics:")
        print(f"  Average Fidelity Plus: {avg_fidelity_Plus:.4f}")
        print(f"  Average Fidelity Minus: {avg_fidelity_Minus:.4f}")
        print(f"  Average Sparsity Edge: {avg_sparsity_edge:.4f}")
        print(f"  Average Sparsity Size: {avg_sparsity_size:.4f}")
