# used to preprocess the data of movielens, latest. The raw data can be found in http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

import csv

PATH = "/Users/morino/Downloads/dataset/ml-latest-small/ratings.csv"
OUTPATH = "/Users/morino/Downloads/dataset/ml-latest-small/ratings-clean.csv"


fid = open(PATH, 'rb');

reader = csv.reader(fid, delimiter = ',');

## skip the head line
reader.next();

reader = list(reader);

item_rind_d = dict();

item_iter = 1;
user_iter = 0;
dataset = list();

for line in reader:
    if(user_iter <= int(line[0])):
        user_iter = int(line[0]);
    if(not (line[1] in item_rind_d)):
        item_rind_d[line[1]] = item_iter;
        item_iter += 1;
    dataset.append(list([int(line[0]), item_rind_d[line[1]], float(line[2])]));


fid.close();



print("USER # {}".format(user_iter));
print("MOVIE # {}".format(len(item_rind_d)));
print("Dataset # {}".format(len(dataset)));


## output the dataset
## training and test division will be later determined
fid = open(OUTPATH, 'wb');
writer = csv.writer(fid, delimiter= ' ');

print("BEGIN WRITE");
for entry in dataset:
    writer.writerow(entry);
print("END WRITE");
    
fid.close();
