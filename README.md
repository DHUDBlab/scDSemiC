# scDSemiC -- Deep Semi-Supervised Clustering based on Self-supervised Learning and Pairwise Constraint Propagation for scRNA-seq data

Single-cell RNA sequencing (scRNA-seq) allows researchers to explore tissue heterogeneity and diversity. Unsupervised
clustering serves a powerful analytical approach for scRNA-seq data, using to identify putative cell types. Previous deep
clustering methods seldom utilize prior knowledge, resulting in clusters that mismatch with the real situation. Recent
semi-supervised clustering achieves improved performance by integrating pairwise constraints. However, the pairwise
constraint information is limited. Here, we propose scDSemiC, a new deep semi-supervised clustering algorithm for
single cell RNA-seq data analysis based on self-supervised learning and pairwise constraint propagation. The proposed
scDSemiC adopts self-supervised feature learning to pretrain the model, and utilizes a pairwise constraint propagation
module to propagate locally limited constraint information, which is critical to guide the clustering process. To validate
the performance of scDSemiC, we compare it with five algorithms on seven real scRNA-seq datasets. The experimental
results demonstrate the proposed algorithm achieve better performance. In addition, the ablation studies on each module
of the algorithm indicate that these modules are complementary to each other and effective in improving the performance
of the proposed algorithm.

Requirements:

Python --- 3.6.8

pytorch -- 1.5.1+cu101

Scanpy --- 1.0.4

faiss --- 1.4.0

Arguments:

n_clusters: number of clusters

n_pairwise: number of pairwise constraints want to generate

gamma: weight of clustering loss

ml_weight: weight of must-link loss

cl_weight: weight of cannot-link loss

Files:

pretrain.py -- pretraining stage of scDSemiC algorithm

fine/fine_CITE.py -- finetune stage of scDSemiC algorithm

generate_matrix.py -- generate matrix to propagate the pairwise constraint

