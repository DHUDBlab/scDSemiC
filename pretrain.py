import os
from time import time

import h5py
import numpy as np
import scanpy.api as sc
import torch
from sklearn import metrics
from sklearn.cluster import KMeans
from preprocess import read_dataset, normalize
from scDSemiC import scDSemiC
from utils import cluster_acc, generate_random_pair

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=8, type=int)
    parser.add_argument('--label_cells', default=0.1, type=float)
    parser.add_argument('--label_cells_files', default='label_selected_cells_1.txt')
    parser.add_argument('--n_pairwise', default=1000, type=int)
    parser.add_argument('--n_pairwise_error', default=0, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='./data/Large_Datasets/Shekhar_mouse_retina_raw_data.h5')
    parser.add_argument('--datasets', default='sh')
    parser.add_argument('--maxiter', default=100, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--ml_weight', default=1., type=float,
                        help='coefficient of must-link loss')
    parser.add_argument('--cl_weight', default=1., type=float,
                        help='coefficient of cannot-link loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/scDCC_p0_1/')
    parser.add_argument('--ae_weight_file', default='AE_weights_p0_1.pth.tar')
    parser.add_argument('--alpha', default=2, type=int)
    parser.add_argument('--beta', default=1, type=int)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])
    data_mat.close()
    args.n_clusters = np.unique(y).shape[0]
    # preprocessing scRNA-seq read counts matrix
    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)

    adata1 = normalize(adata,
                       size_factors=True,
                       normalize_input=True,
                       logtrans_input=True,
                       highly_genes=2000)

    input_size = adata1.n_vars

    print(y.shape)

    if not os.path.exists(args.label_cells_files):
        indx = np.arange(len(y))
        np.random.shuffle(indx)
        label_cell_indx = indx[0:int(np.ceil(args.label_cells * len(y)))]
    else:
        label_cell_indx = np.loadtxt(args.label_cells_files, dtype=np.int)

    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)

    sd = 2.5

    model = scDSemiC(input_dim=adata1.n_vars, z_dim=32, n_clusters=args.n_clusters,
                  encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=sd, gamma=args.gamma,
                  ml_weight=args.ml_weight, cl_weight=args.ml_weight).cuda()
    print(str(model))

    t0 = time()
    model.pretrain_autoencoder(x=adata1.X, X_raw=adata1.raw.X, size_factor=adata1.obs.size_factors,
                                batch_size=args.batch_size, epochs=args.pretrain_epochs,
                                ae_weights=args.ae_weight_file, alpha=args.alpha, beta=args.beta,
                                datasets=args.datasets)
    print('Pretraining time: %d seconds.' % int(time() - t0))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    data, q_all = model.encodeBatch(adata1.X)
    kmeans = KMeans(args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(data.data.cpu().numpy())


    eval_cell_y_pred = np.delete(y_pred, label_cell_indx)
    eval_cell_y = np.delete(y, label_cell_indx)
    acc = np.round(cluster_acc(eval_cell_y, eval_cell_y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(eval_cell_y, eval_cell_y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(eval_cell_y, eval_cell_y_pred), 5)
    print('Evaluating cells: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
    if not os.path.exists(args.label_cells_files):
        np.savetxt(args.label_cells_files, label_cell_indx, fmt="%i")
