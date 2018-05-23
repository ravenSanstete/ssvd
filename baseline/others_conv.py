## we would like to compare with other classical methods with the implementation in the scikit-surprise

import pandas as pd
import surprise as sp
import numpy as np
from surprise.model_selection import cross_validate


## this function converts a dataset in a numpy matrix into a dataframe in pandas' language
def convert_to_df(ds, rating_range):
    ratings_dict = {
        "uid" : list(ds[:, 0].astype(np.int)),
        "vid" : list(ds[:, 1].astype(np.int)),
        "r" : list(ds[:, 2])
    }
    reader = sp.Reader(rating_scale = rating_range);
    df = pd.DataFrame(ratings_dict);
    return sp.Dataset.load_from_df(df[['uid','vid', 'r']], reader);



def load_data(path, r_range):
    train_set = convert_to_df(np.load(path + ".train"), r_range);
    test_set = convert_to_df(np.load(path + ".test"), r_range);
    return train_set.build_full_trainset(), test_set.build_full_trainset().build_testset();
    
    


if __name__ == '__main__':
    PREFIX = "/home/mlsnrs/data/pxd/paper4graduation/paper_exp/dataset/";
    names = [ 'ml-latest-small/ml', 'BX-CSV-Dump/bx'];
    teller = ["MovieLens", "BookCrossing"];
    r_ranges = [(1, 5), (1, 10), (1,5)];
    algos = [sp.SVD(n_factors = 10, biased = False, verbose = True),  sp.NMF(n_factors = 15, verbose = True)];
    algos_names = ['SVD', 'NMF']
    max_epoch = 20;

    for i, name in enumerate(names):
        print("BEGIN {}".format(teller[i]));
        train_set, test_set = load_data(PREFIX + name, r_ranges[i]);
        for j in range(2):
            for epoch in range(1, max_epoch+1):
                avg_rmse = 0.0;
                avg_mae = 0.0;
                for k in range(5):
                    if(j == 0):
                        algo = sp.SVD(n_factors = 10, n_epochs = epoch,biased = False, verbose = False);
                    else:
                        algo = sp.NMF(n_factors = 15, n_epochs = epoch, verbose = False);
                    algo.fit(train_set);
                    preds = algo.test(test_set);
                    avg_rmse += sp.accuracy.rmse(preds, verbose = False);
                    avg_mae += sp.accuracy.mae(preds, verbose = False);
                print("{} EPOCHS {} RMSE {}".format(epoch, algos_names[j], avg_rmse/5.0));
                print("{} EPOCHS {} MAE {}".format(epoch, algos_names[j], avg_mae/5.0));
        print("END {}".format(teller[i]));
