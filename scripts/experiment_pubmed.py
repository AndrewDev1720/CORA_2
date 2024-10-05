import os
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
import sys
from utils.argument import arg_parse_exp_node_pubmed  # Updated for PubMed
from time import time

# Import your utilities
from utils.utils import assemble_EXP_Data, calculate_Metrics, sample_according_to_l_hop_density

# Import your explainer model
from models.explainer_models import NodeExplainerEdgeMulti
from models.gcn import get_model

def print_characteristics(data):
    print(f"NumNodes: {data.num_nodes}")
    print(f"NumEdges: {data.edge_index.size(1)}")  # Edge index is [2, num_edges]
    print(f"NumFeats: {data.num_node_features}")
    print(f"NumClasses: {data.y.max().item() + 1}")
    print(f"NumTrainingSamples: {data.train_mask.sum().item()}")
    print(f"NumValidationSamples: {data.val_mask.sum().item()}")
    print(f"NumTestSamples: {data.test_mask.sum().item()}")

if __name__ == "__main__":
    # Increase print options for large adjacency matrices
    np.set_printoptions(threshold=sys.maxsize)
    
    # Parse experiment arguments for PubMed
    exp_args = arg_parse_exp_node_pubmed()
    print("Arguments:\n", exp_args)
    
    # Load the PubMed dataset
    dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
    
    # Set the device
    device = torch.device('cuda:%s' % exp_args.cuda) if exp_args.gpu else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load the pretrained model
    model_path = exp_args.model_path
    pretrained_model_path = os.path.join(model_path, 'pubmed_gcn_model.pth')
    
    # Initialize the model
    model = get_model(
        input_dim=dataset.num_node_features,  # 500 for PubMed
        hidden_dim=16,                        # Same as Cora
        output_dim=dataset.num_classes         # 3 for PubMed
    ).to(device)
    
    # Load the pre-trained model if it exists
    if os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print(f"Loaded pre-trained model from {pretrained_model_path}")
    else:
        print(f"Pre-trained model not found at {pretrained_model_path}. Please train the model first.")
        sys.exit(1)
    
    model.eval()
    
    # Get the graph data
    data = dataset[0].to(device)
    print_characteristics(data)
    
    # Create the explainer
    explainer = NodeExplainerEdgeMulti(
        base_model=model,
        G_dataset=data,
        args=exp_args,
        test_indices=data.test_mask.nonzero(as_tuple=True)[0]
    )
    
    # Sample nodes according to l-hop density
    listNode, average = sample_according_to_l_hop_density(data, 3, 0.01, 42)
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
        num_nonzero = (masked_adj > exp_args.mask_thresh).sum()
        print(f"Node {node_id}: masked_adj has {num_nonzero} edges above the threshold.")
    
        rows, cols = np.where(masked_adj == 1)
        num_edges_exp = len(rows)
        print(num_edges_exp)
        exp_edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
    
        # Create the explanation subgraph Data object
        exp_subgraph = Data(
            x=x.clone(),
            edge_index=exp_edge_index,
            y=y.clone()
        )
        exp_sub_graph[node_id] = exp_subgraph
    
        # Compute complement edges (edges not in the explanation)
        original_edges_set = set(map(tuple, original_edge_index.T.tolist()))
        exp_edges_set = set(map(tuple, exp_edge_index.T.tolist()))
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
