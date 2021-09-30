import pandas as pd
import csv
import os
import torch
import re
import numpy as np
# pattern = re.compile(r'\d+')

train = 'data/train'
test = 'data/test'

def solve(path):
    files = os.listdir(path)
    for name in files:
        file = os.path.join(path,name)
        df = pd.read_csv(file)
        data = df.to_string().split('\n')
        write_data = []
        for index,x in enumerate(data):
            tmp = x.split(' ')
            tmp = [float(v) for v in tmp if len(v) > 0]
            if (index == 0):
                write_data.append(tmp)
            else:
                write_data.append(tmp[1:])
        f = open(file, 'w', newline='')

        write = csv.writer(f)
        for i in write_data:
            write.writerow(i)
        f.close

def cal(path):
    files = os.listdir(path)
    for name in files:
        file = os.path.join(path, name)
        solve(file)

def write_data(path,num_row):
    f = open(path,'w',newline='')
    x = torch.randn(num_row,22).numpy()
    write = csv.writer(f)
    write.writerows(x)
    f.close()
write_data('data/train/ictal/1.csv',3000)
write_data('data/train/preictal/1.csv',3000)
write_data('data/test/ictal/2.csv',500)
write_data('data/test/preictal/2.csv',500)

# cal(train)
# print('***')
# cal(test)
# print('***')
