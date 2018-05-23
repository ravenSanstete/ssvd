# clean the data from jester, source addr http://www.ieor.berkeley.edu/~goldberg/jester-data/jester-data-1.zip

import xlrd
import csv

PATH = "/Users/morino/Downloads/dataset/jester/jester-data-1.xls";
OUTPATH = "/Users/morino/Downloads/dataset/jester/jester-clean.csv";

book = xlrd.open_workbook(PATH);

sh = book.sheet_by_index(0);
dataset = list();

OFFSET = 10.0; ## to normalize the rating to be positive

print("{0} {1} {2}".format(sh.name, sh.nrows, sh.ncols))

for rx in range(sh.nrows):
    for ry in range(1, sh.ncols):
        val = sh.cell_value(rx, ry);
        if(val <= 11):
            dataset.append(list([rx+1, ry, float(val) + OFFSET]));


print("USER # {}".format(sh.nrows));
print("MOVIE # {}".format(sh.ncols - 1));
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
