from dimod import ConstrainedQuadraticModel, Binary, cqm_to_bqm
from dwave.samplers import SimulatedAnnealingSampler
import numpy as np
import hypernetx as hnx


def hnx_to_laplacian_components(hg: hnx.Hypergraph):
    node_names = sorted(list(hg.nodes))
    edge_names = list(hg.edges)
    node_map = {node: i for i, node in enumerate(node_names)}
    n_v = len(node_names)
    n_e = len(edge_names)
    H = np.zeros((n_v, n_e))
    weights = []

    for j, e_name in enumerate(edge_names):
        nodes_in_edge = hg.edges[e_name]
        for v in nodes_in_edge:
            if v in node_map:
                H[node_map[v], j] = 1
        w = 1.0
        edge_obj = hg.edges[e_name]
        if hasattr(edge_obj, 'weight') and edge_obj.weight is not None:
            w = float(edge_obj.weight)
        weights.append(w)

    W = np.diag(weights)
    delta_e = np.sum(H, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        De_inv_vals = 1.0 / delta_e
        De_inv_vals[~np.isfinite(De_inv_vals)] = 0
    De_inv = np.diag(De_inv_vals)
    d_v = np.dot(H, np.dot(W, np.ones(n_e)))
    with np.errstate(divide='ignore', invalid='ignore'):
        Dv_inv_sqrt_vals = 1.0 / np.sqrt(d_v)
        Dv_inv_sqrt_vals[~np.isfinite(Dv_inv_sqrt_vals)] = 0
    Dv_inv_sqrt = np.diag(Dv_inv_sqrt_vals)

    return H, W, De_inv, Dv_inv_sqrt, node_names


def get_hypergraph_laplacian(H, W, De_inv, Dv_inv_sqrt):
    n_v = H.shape[0]
    I = np.eye(n_v)
    step1 = np.dot(H, np.dot(W, np.dot(De_inv, H.T)))
    Theta = np.dot(Dv_inv_sqrt, np.dot(step1, Dv_inv_sqrt))

    Delta = I - Theta
    return Delta


def qubo_clustering(h, k):
    H, W, De_inv, Dv_inv_sqrt, node_names = hnx_to_laplacian_components(h)
    Delta = get_hypergraph_laplacian(H, W, De_inv, Dv_inv_sqrt)
    cqm = ConstrainedQuadraticModel()
    num_nodes = len(node_names)
    x = {}
    for i, node_name in enumerate(node_names):
        x[i] = [Binary(f"{node_name}_{c}") for c in range(k)]

    objective = 0
    for c in range(k):
        for i in range(num_nodes):
            for j in range(num_nodes):
                coeff = Delta[i, j]
                if abs(coeff) > 1e-9:
                    objective += coeff * x[i][c] * x[j][c]

    cqm.set_objective(objective)
    for i in range(num_nodes):
        cqm.add_constraint(sum(x[i]) == 1, label=f'one_hot_{i}')

    target_size = num_nodes / k
    for c in range(k):
        cqm.add_constraint(sum(x[i][c] for i in range(
            num_nodes)) == target_size, label=f'balance_{c}', weight=0.1)
    bqm, invert = cqm_to_bqm(cqm)
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=100, num_sweeps=1000)
    best_sample = sampleset.first.sample
    original_sample = invert(best_sample)

    res = {}
    for i, node_name in enumerate(node_names):
        for c in range(k):
            if original_sample[f"{node_name}_{c}"] == 1:
                res[node_name] = c
                break
    return res
