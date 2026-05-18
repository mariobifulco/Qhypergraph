from sklearn.cluster import KMeans
import numpy as np


def spectral_hypergraph_clustering(h, num_clusters, weights=None):
    vertices, hyperedges = list(h.nodes), h.edges.elements
    if weights is None:
        weights = {e: 1.0 for e in hyperedges.keys()}

    v_list = sorted(list(vertices))
    v_map = {v: i for i, v in enumerate(v_list)}
    e_list = list(hyperedges.keys())
    n_v = len(v_list)
    n_e = len(e_list)
    h = np.zeros((n_v, n_e))
    for j, e_id in enumerate(e_list):
        for v in hyperedges[e_id]:
            h[v_map[v], j] = 1
    w = np.diag([weights[e] for e in e_list])
    d_v = np.array([sum(weights[e]
                   for e in hyperedges if v in hyperedges[e]) for v in v_list])
    d_v[d_v == 0] = 1e-12
    dv_inv_sqrt = np.diag(1.0 / np.sqrt(d_v))
    d_e = np.array([len(hyperedges[e]) for e in e_list])
    d_e[d_e == 0] = 1e-12
    de_inv = np.diag(1.0 / d_e)

    term1 = dv_inv_sqrt @ h
    term2 = w @ de_inv
    p = term1 @ term2 @ h.T @ dv_inv_sqrt

    _, eigenvectors = np.linalg.eigh(p)
    x = eigenvectors[:, -num_clusters:]
    row_norms = np.linalg.norm(x, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1e-12
    x_normalized = x / row_norms

    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(x_normalized)

    return {v_list[i]: int(labels[i]) for i in range(n_v)}
