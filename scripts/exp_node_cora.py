import os
import numpy as np
import torch
from utils.argument import arg_parse_exp_node_cora  # Import the new argument parser
from models.explainer_models import NodeExplainerEdgeMulti
from models.gcn import get_model  # Assuming you're using get_model to load the GCN model from your pretrained model
from torch_geometric.datasets import Planetoid
from utils.common_utils import Explanation, process_masked_adj
import sys

def print_cora_characteristics(data):
    print(f"NumNodes: {data.num_nodes}")
    print(f"NumEdges: {data.edge_index.size(1)}")  # Edge index is [2, num_edges]
    print(f"NumFeats: {data.num_node_features}")
    print(f"NumClasses: {data.y.max().item() + 1}")
    print(f"NumTrainingSamples: {data.train_mask.sum().item()}")
    print(f"NumValidationSamples: {data.val_mask.sum().item()}")
    print(f"NumTestSamples: {data.test_mask.sum().item()}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1000)
    np.random.seed(0)
    np.set_printoptions(threshold=sys.maxsize)


    # Parse experiment arguments for Cora
    exp_args = arg_parse_exp_node_cora()
    print("argument:\n", exp_args)

    

    # Load the Cora dataset using PyTorch Geometric
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

    # Set the device to GPU or CPU
    # if exp_args.gpu:
    #     device = torch.device('cuda:%s' % exp_args.cuda)
    # else:
    #     device = 'cpu'

    # Load the pretrained model path
    model_path = exp_args.model_path
    pretrained_model_path = os.path.join(model_path, 'cora_gcn_model.pth')

    # Initialize the model with the same architecture as your pretrained model
    model = get_model(
        input_dim=dataset.num_node_features,
        hidden_dim=16,  # Hidden layer dimension, could be modified based on your model
        output_dim=dataset.num_classes  # Number of output classes in Cora
    )

    # Load the state_dict of the pretrained model
    model.load_state_dict(torch.load(pretrained_model_path), map_location=device)
    model.eval()
    model.to(device)

    # Get the first (and only) graph in the dataset
    data = dataset[0].to(device)

    print_cora_characteristics(data)

    # Create the explainer using the loaded model and dataset
    explainer = NodeExplainerEdgeMulti(
        base_model=model,              # The pretrained model
        G_dataset=data,                # The Cora dataset loaded with PyTorch Geometric
        args=exp_args,                 # Arguments (CUDA, dataset, etc.)
        test_indices=data.test_mask.nonzero(as_tuple=True)[0]  # Use test nodes for explanation
    )

    # Run the explainer to generate explanations
    exp_dict, num_dict = explainer.explain_nodes_gnn_stats()

    # Print the explanation results
    # for node_id, masked_adj in exp_dict.items():
    #     print(f"Node {node_id}: Explanation Masked Adjacency")
    #     print(f"shape: {len(masked_adj)}")
    #     print(masked_adj)
    #     print(f"Number of explanations (edges): {num_dict[node_id]}")

    for node_id, masked_adj in exp_dict.items():
        print(f"Node {node_id}: Explanation Masked Adjacency")
        if masked_adj is not None:
            print(f"shape: {masked_adj.size()}")
            print(masked_adj)
            print(f"Number of explanations (edges): {num_dict[node_id]}")
        else:
            print("No explanation was generated for this node.")
    
    explanations = {}
    mask_threshold = exp_args.mask_thresh

    for node_id, masked_adj in exp_dict.items():
        # Process masked_adj to get edge_index and edge_mask
        edge_index, edge_mask = process_masked_adj(masked_adj, mask_threshold)

        # Create the explanation object
        explanation = Explanation(
            edge_index=edge_index,
            edge_mask=edge_mask,
            x=data.x,
            y=data.y[node_id]
        )

        explanations[node_id] = explanation
