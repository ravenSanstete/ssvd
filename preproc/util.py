import csv
import numpy as np

def read_csv(path):
    fid = open(path, 'rb');
    reader = csv.reader(fid, delimiter = ' ');
    ds = list(reader);
    fid.close();
    return ds;

def write_csv(ds, outpath):
    fid = open(outpath, 'wb');
    writer = csv.writer(fid, delimiter= ' ');

    print("BEGIN WRITE");
    for entry in ds:
        writer.writerow(entry);
    print("END WRITE");
    fid.close();


## split the dataset into training and testing set (ratio := # of training/ # of ds)
def split_ds(ds, ratio):
    print("BEGIN shuffle twice");
    np.random.shuffle(ds);
    np.random.shuffle(ds);
    print("END shuffle twice");
    train_num = int(len(ds)*ratio);
    return np.array(ds[:train_num], dtype = np.float32), np.array(ds[train_num+1:], dtype = np.float32);

def split_csv(path, ratio, out_prefix):
    print("BEGIN PROCESSING {}".format(path));
    train_set, test_set = split_ds(read_csv(path), ratio);
    fid_1, fid_2 = open(out_prefix+".train", 'wb'), open(out_prefix+".test", 'wb');
    np.save(fid_1, train_set);
    np.save(fid_2, test_set);
    print("# of Train Set {}".format(len(train_set)));
    print("# of Test Set {}".format(len(test_set)));
    fid_1.close();
    fid_2.close();
    print("END PROCESSING {}".format(path));

if __name__ == '__main__':
    PREF = "/Users/morino/Downloads/dataset/";
    RAW_NAMEs = ['BX-CSV-Dump/BX-Clean.csv', "jester/jester-clean.csv", 'ml-latest-small/ratings-clean.csv', 'douban/douban-clean.csv'];
    OUT_PREFs = ['BX-CSV-Dump/bx', 'jester/jester', 'ml-latest-small/ml', 'douban/douban'];
    ratio = 0.8;
    for i in range(3, len(RAW_NAMEs)):
        split_csv(PREF + RAW_NAMEs[i], ratio, PREF + OUT_PREFs[i]);
