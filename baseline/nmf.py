# this implements the NMF baseline for comparison with the application of the implemented version in
# sklean http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

## this implementation is not relevant, which is poor in performance, which is not useful as a baseline
import numpy as np
from sklearn.decomposition import NMF

from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix, coo_matrix


## to convert the original dataset into some sparse matrix
def convert_to_sparse_mat(ds):
    return coo_matrix((ds[:, 2], (ds[:,0].astype(np.int), ds[:,1].astype(np.int))));


## load the data and return the training data as a matrix
def load_data(path):
    train_set = np.load(path + ".train");
    test_set = np.load(path + ".test");
    return convert_to_sparse_mat(train_set), test_set;



## input the sparse matrix and output the matrix W and H
# r the dimension of the latent feature
def nmf(sp_mat, r):
    model = NMF(n_components = r, init = 'nndsvd', tol= 0.00005, max_iter = 500, verbose = True, alpha = 0.05);
    return model.fit_transform(sp_mat), (model.components_).T;
    

def predict(test_set, W, H):
    pred = list();
    for row in test_set:
        pred.append(np.dot(W[int(row[0]),:], H[int(row[1]), :]));
    return pred;





if __name__ == '__main__':
    PREFIX = "/Users/morino/Downloads/dataset/";
    names = [ 'ml-latest-small/ml', 'BX-CSV-Dump/bx', 'jester/jester'];
    teller = ["MovieLens", "BookCrossing", "Jester"];

    R = 10;

    for i, name in enumerate(names):
        print("BEGIN {}".format(teller[i]));
        train_sp_mat, test_set = load_data(PREFIX + name);
        W, H = nmf(train_sp_mat, R);
        preds = predict(test_set, W, H);
        print("{} NMF RMSE: {}".format(teller[i], np.sqrt(mean_squared_error(test_set[:, 2], preds))));
        print("END {}".format(teller[i]));
    
