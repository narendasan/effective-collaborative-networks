import torch

def convert_sd_to_adj(name, state_dict):
    keys = state_dict.keys()
    weights = []
    biases = []
    for k in keys:
        if "weight" in k:
            weights.append(k)
        elif "bias" in k:
            biases.append(k)
    
    edges = []
    in_dim_start = 0
    for w in weights:
        edge_weights = state_dict[w].numpy()
        edge_weights = edge_weights.T
        for i in range(edge_weights.shape[0]):
            in_dim = in_dim_start + i
            for j in range(edge_weights.shape[1]):
                out_dim = in_dim_start + edge_weights.shape[0] + j
                edges.append((in_dim, out_dim, edge_weights[i][j]))
        in_dim_start = in_dim_start + edge_weights.shape[0]
                
    open("data/ann/" + name + '.adj', 'w').write('\n'.join('%s %s %s' % x for x in edges))