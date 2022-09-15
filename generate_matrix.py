import os

import scanpy.api as sc
import torch.optim as optim
import torch
from scipy import sparse
from scipy.io import savemat

from preprocess import read_dataset, normalize
from scDSemiC import scDSemiC
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
checkpoint = torch.load(
    './results/pretrain/Shekhar_mouse_retina_raw_data_model.pth.tar')
data_mat = h5py.File('./scDCC/data/Large_Datasets/Shekhar_mouse_retina_raw_data.h5')
x = np.array(data_mat['X'])
genes = x.shape[1]
y = np.array(data_mat['Y'])
# embedding = np.array(data_mat['ADT_X'])
x_y = np.insert(x, x.shape[1], values=y, axis=1)
x_y1 = x_y[x_y[:, x.shape[1]].argsort()]
a = x_y[:, x.shape[1]].argsort()
x1 = x_y1[:, 0:genes]
# embedding_y = np.insert(embedding, embedding.shape[1], values=y, axis=1)
# embedding_y1 = embedding_y[embedding_y[:, embedding.shape[1]].argsort()]
# b = embedding_y[:, embedding.shape[1]].argsort()
# embedding1 = embedding_y1[:, 0:embedding.shape[1]]
y.sort()

data_mat.close()
adata = sc.AnnData(x1)
adata.obs['Group'] = y

adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)
# sc.pp.filter_cells(adata, min_genes=800)
adata1 = normalize(adata,
                   size_factors=True,
                   normalize_input=True,
                   logtrans_input=True,
                   highly_genes=2000)

model = scDSemiC(input_dim=adata1.n_vars, z_dim=32, n_clusters=8,
              encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=2.5, gamma=1.).cuda()
print(model)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, amsgrad=True)
model.load_state_dict(checkpoint['ae_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
X = adata1.X
X = torch.tensor(X).cuda()
data, _, _, _, _, _ = model.forward(X)
data1 = data[0:data.shape[0] // 2, :]
data2 = data[data.shape[0] // 2:, :]

W = generate_W(data)
W1 = generate_W(data1)
W2 = generate_W(data2)
# savemat("./MaW.mat", {'W': W})
T = W.todense()
T1 = W1.todense()
T2 = W2.todense()
# W1 = generate_W(X)
# T1 = W1.todense()
# print(T)
# np.savetxt('./shW1.csv', T1, delimiter=',', fmt="%.2f")
# np.savetxt('./shW2.csv', T2, delimiter=',', fmt="%.2f")
# print("saved W")
# label_cells_files = 'label_selected_cells_1.txt'
indx = np.arange(len(y))
indx1 = indx[0:indx.shape[0] // 2]
indx2 = indx[indx.shape[0] // 2:]
np.random.shuffle(indx)
np.random.shuffle(indx1)
np.random.shuffle(indx2)
label_cell_indx = indx[0:int(np.ceil(0.1 * len(y)))]
label_cell_indx1 = indx1[0:int(np.ceil(0.1 * len(y)//2))]
label_cell_indx2 = indx2[0:int(np.ceil(0.1 * len(y)//2))]
# label_cell_indx = np.loadtxt(label_cells_files, dtype=np.int)
ml_ind11, ml_ind21, cl_ind11, cl_ind21, error_num1 = generate_random_pair(y, label_cell_indx1, 5000,
                                                                     0)
ml_ind12, ml_ind22, cl_ind12, cl_ind22, error_num2 = generate_random_pair(y, label_cell_indx2, 5000,
                                                                     0)
# ml_ind1, ml_ind2, cl_ind1, cl_ind2 = generate_random_pair_from_proteins(embedding1, 5000,
#                                                                                 0.005, 0.95)
# markers = np.loadtxt("./data/CITE_PBMC/adt_CD_normalized_counts.txt", delimiter=',')
# ml_ind1_2, ml_ind2_2, cl_ind1_2, cl_ind2_2 = generate_random_pair_from_CD_markers(markers, 5000,
#                                                                                   0.3, 0.7, 0.3, 0.85)
# print("Must link paris: %d" % ml_ind1_1.shape[0])
# print("Cannot link paris: %d" % cl_ind1_1.shape[0])
# ml1, ml2, cl1, cl2 = generate_pair(y, label_cell_indx)
# print(len(ml_ind1_1))
# print(len(cl_ind1_1))
# print(len(ml_ind1_2))
# print(len(cl_ind1_2))
# print(len(ml1))
# print(len(cl1))
# # for i in range(len(ml_ind1)):
# #     a = np.where(ml_ind1 == 353)
# #     print(a)
# #     break
# # m1 = ml_ind1.sort()
# # m2 = ml_ind2.sort()
# print(ml_ind1)
# print(ml_ind2)
# y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors, y=y,
#                                batch_size=256, num_epochs=2000,
#                                ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2,
#                                update_interval=1, tol=0.001, save_dir='results/scDCC_p0_1/')
# ml_ind1 = np.append(ml_ind1_1, ml_ind1_2)
# ml_ind2 = np.append(ml_ind2_1, ml_ind2_2)
# cl_ind1 = np.append(cl_ind1_1, cl_ind1_2)
# cl_ind2 = np.append(cl_ind2_1, cl_ind2_2)
P1, S1, D1 = construct_P(T1, ml_ind11, ml_ind21, cl_ind11, cl_ind21)
F1 = construct_F(T1, ml_ind11, ml_ind21, cl_ind11, cl_ind21)
P2, S2, D2 = construct_P(T2, ml_ind12, ml_ind22, cl_ind12, cl_ind22)
F2 = construct_F(T2, ml_ind12, ml_ind22, cl_ind12, cl_ind22)
# sP = sparse.csr_matrix(P)
# sD = sparse.csr_matrix(D)
# sF = sparse.csr_matrix(F0)
# savemat("./Ma1000P.mat", {'P': sP})
# print("saved P")
# savemat("./Ma1000D.mat", {'D': sD})
# print("saved D")
# savemat("./Ma1000F.mat", {'F': sF})
# print("saved F")
# P1, S1, D1 = construct_P(T, ml1, ml2, cl1, cl2)
# F1 = construct_F(T, ml1, ml2, cl1, cl2)
# #
np.savetxt('./sh5000P1.csv', P1, delimiter=',', fmt="%d")
np.savetxt('./sh5000P2.csv', P2, delimiter=',', fmt="%d")
print("saved P")
# np.savetxt('./mouse4000S.csv', S, delimiter=',', fmt="%.2f")
np.savetxt('./sh5000D1.csv', D1, delimiter=',', fmt="%d")
np.savetxt('./sh5000D2.csv', D2, delimiter=',', fmt="%d")
print("saved D")
np.savetxt('./sh4000F1.csv', F1, delimiter=',', fmt="%d")
np.savetxt('./sh4000F2.csv', F2, delimiter=',', fmt="%d")
print("saved F")
#
# # P1, S1, D1 = construct_P(T1, ml_ind1, ml_ind2, cl_ind1, cl_ind2)
# print(P, S, D)
# print(ml_ind1, ml_ind2, cl_ind1, cl_ind2, error_num)
