import random
import os
import numpy as np
# import yaml
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, AmazonProducts, FacebookPagePage, KarateClub
from torch_geometric.transforms import NormalizeFeatures
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx,k_hop_subgraph

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def dataset_func(config, random_seed):
    # if config['data_name'] == 'Facebook' and config['exp_name']=='fair':
    #     return load_fair_data()
    # if config['data_name'] == 'facebook100' and config['exp_name']=='fair':
    #     return torch.load('./datasets/facebook100/data.pt')
    # elif config['data_name'] == 'aml' and config['exp_name']=='fair':
    #     return torch.load('./datasets/aml/data.pt')
    # elif config['data_name'] == 'german' and config['exp_name']=='fair':
    #     return torch.load('./datasets/german/data.pt')
    # elif config['data_name'] == 'amazon':
    #     return AmazonProducts(root="./datasets/amazon")[0]
    
    data_dir = "./datasets"
    data_name = config['data_name']
    data_size = config['data_size']
    num_class = config['output_dim']
    num_test = config['num_test']
    random_seed = config['random_seed']
    os.makedirs(data_dir, exist_ok=True)
    set_seed(random_seed)
    num_train_per_class = (data_size - num_test)//num_class
    if data_name in ['Cora', 'PubMed','CiteSeer']:
        data = Planetoid(root=data_dir, name=data_name, split='random', num_train_per_class=num_train_per_class, num_val=0, num_test=num_test)[0]
    elif data_name == 'amazon':
        amazon_dir = os.path.join(data_dir, 'amazon')
        os.makedirs(amazon_dir, exist_ok=True)
        data = AmazonProducts(root=data_dir)[0]
        num_nodes = data.num_nodes
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[:int(0.8 * num_nodes)] = True  # 80% for training

        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask[int(0.8 * num_nodes):int(0.9 * num_nodes)] = True  # 10% for validation

        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[int(0.9 * num_nodes):] = True  # 10% for testing
        save_path = os.path.join(amazon_dir, "Amazon_processed.pt")
        torch.save(data, save_path)  # save data


    elif data_name == 'facebookpagepage':
        facebookpp_dir = os.path.join(data_dir, 'facebookpagepage')
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

    elif data_name == 'Karate':
        karate_dir = os.path.join(data_dir, 'Karate')
        os.makedirs(karate_dir, exist_ok=True)
        data = KarateClub(transform=NormalizeFeatures())[0]
        num_nodes = data.num_nodes

        # 为了保证每个类别都有足够的样本，手动指定训练、验证和测试集
        torch.manual_seed(random_seed)
        perm = torch.randperm(num_nodes)

        num_train = int(0.6 * num_nodes)
        num_val = int(0.2 * num_nodes)
        num_test = num_nodes - num_train - num_val

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[perm[:num_train]] = True

        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask[perm[num_train:num_train + num_val]] = True

        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[perm[num_train + num_val:]] = True

        save_path = os.path.join(karate_dir, "karate_processed.pt")
        torch.save(data, save_path)

    return data


#Determine if it's factual and counterfactual exp
#Assemble EXP subgraph, complementary graph
#Inference
#Calculate Fidelity-, Fidelity+, Sparsity, diveristy(tbf)
def assemble_EXP_Data(explanation, edge_threshold, feature_threshold=None):
    edge_index = explanation.edge_index
    edge_mask = explanation.edge_mask
    x = explanation.x
    y = explanation.target

    #Filter out important edges
    important_edges_masks = edge_mask > edge_threshold
    important_edge_indices = important_edges_masks.nonzero(as_tuple=True)[0]

    #Build exp data
    exp_edge_index = edge_index[:, important_edge_indices]
    

    exp_subgraph = Data(
        x=x.clone(),
        edge_index = exp_edge_index,
        y = y
    )

    comp_edges_masks = ~important_edges_masks
    comp_edge_indices = comp_edges_masks.nonzero(as_tuple=True)[0]
    comp_edge_index = edge_index[:,comp_edge_indices]

    #Build comp data
    comp_subgraph = Data(
        x=x.clone(),
        edge_index=comp_edge_index,
        y=y
    )

    #check if there is node feature mask
    if hasattr(explanation, 'node_feat_mask') and explanation.node_feat_mask is not None:
        node_feat_mask = explanation.node_feat_mask # shape:[num_features]

        if feature_threshold is None:
            feature_threshold = node_feat_mask.mean().item()
            #if no threshold designated then use mean as the threshold

        important_features_mask = node_feat_mask > feature_threshold

        exp_subgraph.x[:, ~important_features_mask] = 0 # unimportant feature now 0
        comp_subgraph.x[:, important_features_mask] = 0 # unimportant feature now 0


    return exp_subgraph, comp_subgraph

#Visualize exp & Induced subgraph
def visual_Exp_Subgraph(exp_subgraph, node_idx=None, remove_isolates=True,font_size=5):
    G = to_networkx(exp_subgraph, to_undirected=True)

    if remove_isolates:
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)

    plt.figure(figsize=(8,8))

    if node_idx is not None:
        #if node_idx is not list, change it to list
        if not isinstance(node_idx,(list, set, tuple)):
            node_idx = [node_idx]
        node_colors = []
        for node in G.nodes():
            if node in node_idx:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
    else:
        node_colors = 'lightblue'

    labels = {}
    for node in G.nodes():
        if node_idx is not None and node in node_idx:
            labels[node] = str(node)
        else:
            labels[node] = ''

    nx.draw_networkx(G, with_labels=True, node_size=20, node_color= node_colors, labels=labels,font_size=font_size)
    plt.title('Explanation Subgraph')
    plt.show()



"""
only use for our method:
we don't care if baselines can give factual & counterfactual results
Our definition of validity: A graph, covers at least a test node, which it does, as we don't delete nodes, we simply use edge mask
and for at least one test node, or more, it is factual or counterfactual
"""
def is_Valid(original_graph, exp_subgraph, comp_subgraph, model, node_idx )->bool:
    model.eval()
    #make node_idx a list
    if not isinstance(node_idx, (list, set, tuple)):
        node_idx = [node_idx]

    node_idx = torch.tensor(node_idx, dtype=torch.long)

    with torch.no_grad():
        original_out = model(original_graph.x, original_graph.edge_index)
        original_pred = original_out[node_idx].argmax(dim=1)

    #inference on exp_subgraph
    with torch.no_grad():
        exp_out = model(exp_subgraph.x, exp_subgraph.edge_index)
        exp_pred = exp_out[node_idx].argmax(dim=1)

    #inference on comp_subgraph
    with torch.no_grad():
        comp_out = model(comp_subgraph.x, comp_subgraph.edge_index)
        comp_pred = comp_out[node_idx].argmax(dim=1)

    factual_array = (exp_pred == original_pred)
    counterfactual_array = (comp_pred != original_pred)

    # factual = factual_array.all().item()
    # counterfactual = counterfactual_array.all().item()

    factual = factual_array.any().item()
    counterfactual = counterfactual_array.any().item()

    # if factual:
    #     print("explanation is factual")
    # else:
    #     print("explanation is not factual")
    
    # if counterfactual:
    #     print("explantion is counterfactual")
    # else:
    #     print('explanation is not counterfactual')

    if factual or counterfactual:
        # print("explanation is valid")
        return True
    else:
        # print("explanation is invalid")
        return False

def calculate_Metrics(original_graph, exp_subgraph, comp_subgraph, model, node_idx):
    """
    use this when the explainer returns one unioned mask for the entire set of test nodes
    """
    model.eval()
    with torch.no_grad():
        logits_original = model(original_graph.x, original_graph.edge_index)
        probs_original = torch.softmax(logits_original, dim=1)
        y_true = original_graph.y

        f_Gi_yi = probs_original[node_idx, y_true[node_idx]]
        #For each node, retrieve the predicted probability on the node's real label

        logits_comp = model(comp_subgraph.x, comp_subgraph.edge_index)
        probs_comp = torch.softmax(logits_comp, dim=1)
        f_Gi_comp_yi = probs_comp[node_idx, y_true[node_idx]]

        logits_exp = model(exp_subgraph.x, exp_subgraph.edge_index)
        probs_exp = torch.softmax(logits_exp, dim=1)
        f_Gi_exp_yi = probs_exp[node_idx, y_true[node_idx]]

        fidelity_Plus = f_Gi_yi - f_Gi_comp_yi
        # fidelity_Plus = fidelity_Plus.mean().item()

        fidelity_Minus = f_Gi_yi - f_Gi_exp_yi
        # fidelity_Minus = fidelity_Minus.mean().item()

        #edge_wise Sparsity
        num_edge_exp = exp_subgraph.edge_index.size(1)
        num_edge_origin = original_graph.edge_index.size(1)
        sparsity_edge = 1 - num_edge_exp/num_edge_origin

        #size_wise sparsity
        num_node_exp = exp_subgraph.edge_index.unique().size(0)
        num_node_origin = original_graph.num_nodes
        size_exp = num_edge_exp + num_node_exp
        size_origin = num_edge_origin + num_node_origin
        sparisty_size = 1 - size_exp/size_origin

    return fidelity_Plus, fidelity_Minus, sparsity_edge, sparisty_size


"""
def calculate_Metrics_Iter(): Use this when the baseline explainer gives one explanation for one node at a time
we simply pass in a list of explanations, and its counterparts
and we normalize them
"""
def calculate_Metrics_Iter(original_graph, exp_subgraphs, comp_subgraphs, model, node_idx):
    model.eval()
    #ensure the node_idx is a tensor
    if not isinstance(node_idx, torch.Tensor):
        node_idx = torch.tensor(node_idx, dtype=torch.long)
    
    fidelity_Plus_list = []
    fidelity_Minus_list = []
    #also sparsity
    sparsity_list = []
    sparsity_list_size = []
    #conciseness
    all_exp_edges = set()

    with torch.no_grad():
        #Inference on the original graph & retrieve the probs
        logits_original = model(original_graph.x, original_graph.edge_index)
        probs_original = torch.softmax(logits_original, dim=1)
        y_true = original_graph.y[node_idx]
        #the true labels set for node_idx

        #For each node, retrive the predicted probability on the node's real label
        f_Gi_yi = probs_original[node_idx, y_true]

        #Iterate over each node and its corresponding exp, comp graph
        for i, node in enumerate(node_idx):
            #comp graph inference
            comp_subgraph = comp_subgraphs[i]
            logits_comp = model(comp_subgraph.x, comp_subgraph.edge_index)
            probs_comp = torch.softmax(logits_comp, dim=1)
            f_Gi_comp_yi = probs_comp[node,y_true[i]]

            #exp graph inference
            exp_subgraph = exp_subgraphs[i]
            logits_exp = model(exp_subgraph.x, exp_subgraph.edge_index)
            probs_exp = torch.softmax(logits_exp, dim=1)
            f_Gi_exp_yi = probs_exp[node, y_true[i]]

            #Fidelity Plus: Compare original graph with complement graph
            fidelity_Plus = (f_Gi_yi[i].item() - f_Gi_comp_yi)
            fidelity_Plus_list.append(fidelity_Plus)

            #Fidelity Minus: Compare original graph with explanation subgraph
            fidelity_Minus = (f_Gi_yi[i].item() - f_Gi_exp_yi)
            fidelity_Minus_list.append(fidelity_Minus)

            num_edge_exp = exp_subgraph.edge_index.size(1)
            num_node_exp = exp_subgraph.edge_index.unique().size(0)
            num_edge_origin = original_graph.edge_index.size(1)
            num_node_origin = original_graph.num_nodes

            size_exp = num_node_exp + num_edge_exp
            size_origin = num_edge_origin + num_node_origin

            sparsity = 1 - num_edge_exp / num_edge_origin
            sparsity_list.append(sparsity)

            sparsity_size = 1 - size_exp / size_origin
            sparsity_list_size.append(sparsity_size)

            exp_edges = exp_subgraph.edge_index.t().tolist()
            for edge in exp_edges:
                all_exp_edges.add(tuple(edge))


        fidelity_Plus_mean = sum(fidelity_Plus_list) / len(fidelity_Plus_list)
        fidelity_Minus_mean = sum(fidelity_Minus_list) / len(fidelity_Minus_list)
        sparsity_mean = sum(sparsity_list) / len(sparsity_list)
        sparsity_size_mean = sum(sparsity_list_size) / len(sparsity_list_size) #size-wise: meaning |edge| + |node|

        conciseness = len(all_exp_edges)

    return fidelity_Plus_mean, fidelity_Minus_mean, sparsity_mean, sparsity_size_mean, conciseness





def induce_k_hop(data, node_idx, k):
    if not node_idx:
        raise ValueError("at least provide one node id")
    if isinstance(node_idx, list):
        node_idx = torch.tensor(node_idx, dtype=torch.long)
    elif isinstance(node_idx, torch.Tensor):
        node_idx = node_idx.long()
    else:
        raise TypeError("node id must be a list or a torch tensor")
    
    subset,edge_index, inv, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=k,
        edge_index=data.edge_index,
        relabel_nodes=False # keep the original node indices        
    )
    full_edge_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
    full_edge_mask[edge_mask] = True

    return full_edge_mask

#input k, retain edges between k : k+1 layer, e.g. input k = 1, retain the edges between 1,2 layer
#i.e. the exact (k+1)-layer neighbor
#1-hop: induce_k_hop(data, node_idx, k)
def Onion_layer(data, node_idx, k):
    if not node_idx:
        raise ValueError("at least provide one node id")
    if isinstance(node_idx, list):
        node_idx = torch.tensor(node_idx, dtype=torch.long)
    elif isinstance(node_idx, torch.Tensor):
        node_idx = node_idx.long()
    else:
        raise TypeError("node id must be a list or a torch tensor")
    
    #induce k - hop subgraph
    _, _, _, edge_mask_k = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=k,
        edge_index=data.edge_index,
        relabel_nodes=False #keep the original node idx
    )
    #induce k+1 - hop subgraph
    _, _, _, edge_mask_k_plus_1 = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=k+1,
        edge_index=data.edge_index,
        relabel_nodes=False
    )

    #create an edge mask the same size as the original edge_index
    edge_mask = torch.zero(data.edge_index.size(1), dtype=torch.bool)
    edge_mask[edge_mask_k_plus_1] = True
    edge_mask[edge_mask_k] = False
    
    return edge_mask


def sample_according_to_average_degree(data, sample_percentage, random_seed):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_edges = edge_index.size(1) #num of columns in edge index
    if random_seed is not None:
        torch.manual_seed(random_seed)

    average_degree = (2 * num_edges) / num_nodes

    degrees = torch.zeros(num_nodes, dtype=torch.int)
    for edge in edge_index.t():
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    
    print(f"Average Degree:{average_degree}")

    #find the nodes whose degrees are close to the average
    threshold = 0.1 * average_degree
    close_to_average = torch.nonzero(torch.abs(degrees - average_degree) <= threshold).view(-1)

    num_samples = max(1, int(sample_percentage * num_nodes))  # at least choose one node
    if len(close_to_average) > 0:
        sampled_nodes = close_to_average[torch.randperm(len(close_to_average))[:num_samples]]
    else:
        sampled_nodes = torch.tensor([])  # if no such node
    return sampled_nodes, average_degree

def sample_according_to_l_hop_density(data, L, sample_percentage, random_seed):
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    if random_seed is not None:
        torch.manual_seed(random_seed)

    # 1. keep trach of every nodes' L-hop density
    densities = torch.zeros(num_nodes, dtype=torch.float)
    
    for node in range(num_nodes):
        # obtain nodes' L-hop subgraph
        subset, _, _, edge_mask = k_hop_subgraph(node, L, edge_index, relabel_nodes=False)
        num_subgraph_edges = edge_mask.sum().item()  # num of edges in L-hop
        num_subgraph_nodes = len(subset)  # num of nodes in L-hop
        # Calculate Density
        if num_subgraph_nodes > 1: 
            density = num_subgraph_edges / num_subgraph_nodes
        else:
            density = 0  # if only one node，density 0
        densities[node] = density

    average_density = densities.mean().item()
    print(f"Average L-hop Density: {average_density}")

   
    threshold = 0.1 * average_density  # ±10%
    close_to_average = torch.nonzero(torch.abs(densities - average_density) <= threshold).view(-1)

   
    num_samples = max(1, int(sample_percentage * num_nodes))  # choose at least one
    if len(close_to_average) > 0:
        sampled_nodes = close_to_average[torch.randperm(len(close_to_average))[:num_samples]]
    else:
        sampled_nodes = torch.tensor([])  # if no such node

    return sampled_nodes, average_density