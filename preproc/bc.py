## to process the data from the source http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip
import csv

PATH = "/Users/morino/Downloads/dataset/BX-CSV-Dump/BX-Book-Ratings.csv";
OUTPATH = "/Users/morino/Downloads/dataset/BX-CSV-Dump/BX-Clean.csv";

fid = open(PATH, 'rb');

reader = csv.reader(fid, delimiter = ';');

## omit the first line
reader.next();

## do some re-index, use the two extra dict
isbn_rind_d = dict();
uid_rind_d = dict();
isbn_iter = 1
uid_iter = 1
dataset = list();

for line in reader:
    ## remove the implicit rating, the research on the rating matrix
    if(int(line[2]) != 0):
        if(not (line[0] in uid_rind_d)):
            uid_rind_d[line[0]] = uid_iter;
            uid_iter += 1;
        if(not (line[1] in isbn_rind_d)):
            isbn_rind_d[line[1]] = isbn_iter;
            isbn_iter += 1;
        dataset.append(list([uid_rind_d[line[0]], isbn_rind_d[line[1]], float(line[2])]))
        
print("USER # {}".format(len(uid_rind_d)));
print("BOOK # {}".format(len(isbn_rind_d)));
print("Dataset # {}".format(len(dataset)));
fid.close();

## output the dataset
## training and test division will be later determined
fid = open(OUTPATH, 'wb');
writer = csv.writer(fid, delimiter= ' ');

print("BEGIN WRITE");
for entry in dataset:
    writer.writerow(entry);
print("END WRITE");
    
fid.close();








