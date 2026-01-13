# Quantum4Hypergraphs

The objective of this project is to use quantum technologies in order to solve the Normalized Min Cut problem for hypergraphs, originally proposed in [Learning with Hypergraphs: Clustering,
Classification, and Embedding](https://proceedings.neurips.cc/paper_files/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf) by Zhou et al.

We aim to:

- Convert the original optimization problem in a QUBO-compatible form;
- Use the QUBO problem to perform quantum annealing;
- Use the produced embedding to classify standard dataset(s);
- Compare the results with the heuristic solution proposed by the original paper.
