#!/usr/bin/env python3

import sys
import numpy
import pandas

filename = 'GlyNet f_size=3 n_hidden_layers=2 weight_decay=0.0.npz'
filename = '1-GlyNet.npz'
filename = sys.argv[1]


with numpy.load(filename, allow_pickle = False) as data_file:
    parameters = data_file['parameters']
    idx        = data_file['index']           # list of row names - molecules
    cols       = data_file['columns']         # list of column names - dataset IDs
    atom_ref   = pandas.DataFrame(data_file['act_data'], index = idx, columns = cols)
    atom_pred  = pandas.DataFrame(data_file['pred_data'], index = idx, columns = cols)
    atom_fold  = pandas.DataFrame(data_file['fold_data'], index = idx, columns = ['Fold'])
    input_data = pandas.DataFrame(data_file['input_data'], index = idx)  #, columns = cols)
    #monitor_data = data_file['monitor_data']


print(parameters)
#print('Monitor Data:', monitor_data)

atom_ref.to_csv('atom_ref.txt', sep = '\t')
atom_pred.to_csv('atom_pred.txt', sep = '\t')
atom_fold.to_csv('atom_fold.txt', sep = '\t')
atom_ref.to_csv('input_data.txt', sep = '\t')


print(atom_ref)
print(atom_pred)
#print(atom_fold)

print(atom_pred.index)

# get a single concentration slice of the reference data
x = atom_ref[ ['0.1 ' == x[:4]  for x in atom_ref.index] ]
print((x >= 0).sum())
