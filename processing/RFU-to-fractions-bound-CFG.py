#!/usr/bin/env python3


import math
import re
import sys
import numpy
import pandas
import sklearn.neighbors

def read_CFG_table(filename):
    # read in the full CFG table
    cfg_full = pandas.read_csv(filename, sep = '\t')

    # calculate RFUs from the signal and background
    cfg_full['RFU'] = cfg_full['Signal.Mean'] - cfg_full['Background.Mean']
    
    # calculate statistics over the replicates
    array = cfg_full.groupby(['ObjId', 'GlycanID']
                ).aggregate({'RFU': ('mean', 'std' , 'size')}
                ).xs('RFU', axis = 1, level = 0
                ).reset_index()
    
    # set-up an index
    array = array.set_index(['ObjId', 'GlycanID'])

    # only keep the mean statistics - rename the means to RFU
    array = array.drop(['std', 'size'], axis = 1)
    array = array.rename({'mean': 'RFU'}, axis = 1)
    
    return array
    # set-up an index


def convert_RFU_to_f(cfg, samples):
    # convert a column of RFU to f by rescaling with parameters from samples
    
    #join the table so we have the base and maximum RFUs for rescaling
    cfg = samples.join(cfg)
    cfg['f'] = ((cfg['RFU'] - cfg['Base']) / (cfg['Maximum'] - cfg['Base']))
    cfg['f'] = cfg['f'].clip(0.0, 1.0)
    return cfg


def interpolate_series(series, conc_frame):
    # interpolate data from series to the datapoints in conc_frame

    # may be duplicate concentrations - merge them by taking mean of f
    series = series.groupby(level = [0, 1]).mean()
    series = series.droplevel(level = 'GroupNum')

    #find the highest concentration we have data for
    max_conc = series.index.get_level_values('Concentration').max()

    # merge the two dataframes together
    interp_series = pandas.concat([series, conc_frame])
    ###interp_series = series     # uncomment to prevent interpolants
    interp_series.index = interp_series.index.set_names('Concentration')

    # the conc_frame may have introduced duplicate concentrations
    # they are removed with groupby().min()
    interp_series = interp_series.groupby(level = 'Concentration').min()

    # now interpolate the numbers between the indices
    interp_series = interp_series.interpolate('values')

    # find entries above the maximum known concentration - and delete
    mask = interp_series.index <= max_conc
    interp_series = interp_series.loc[mask]
    
    # restrict the output to only concentrations in the conc_frame
    x = conc_frame[ [] ].join(interp_series, how = 'inner')
    return x

def rescale_rfu(rfu_data):
    # convert RFU to f and extract the appropriate subset

    # perform the KDE and find the baseline
    rfu_curve = [(0.0 + 4.8/99*x,) for x in range(100)]
    kde_curve = []

    kde2 = sklearn.neighbors.KernelDensity(kernel = 'gaussian',
                    bandwidth = 0.2).fit([(math.log10(x),)
                                          for x in rfu_data  if x > 0.0])
    curve2 = numpy.exp(kde2.score_samples(rfu_curve))

    max_density = max([o  for v, o in zip(rfu_curve, curve2)  if v[0] < 3.5])
    max_idx = list(curve2).index(max_density)
    baseline = 10 ** rfu_curve[max_idx][0]
    topline  = 65535
    kde_curve.append(curve2)

    f = (rfu_data - baseline) / (topline - baseline)
    f = f.clip(lower = 0.0, upper = 1.0)

    return f


def convert_to_f(rfu_data):
    # convert RFU to f and extract the appropriate subset
    f_data = pandas.DataFrame(columns = ['Glycan', 'Lectin',
                                       'Concentration', 'Fraction_Bound'])
    print(f_data)
    zeros = set()

    for objId in rfu_data.index.drop_duplicates():
        subset = rfu_data.loc[objId]
        
        # extract column of data and rescale it
        rfu_column = subset['RFU']
        
        data_chunk = subset[['GlycanID', 'Lectin']].copy()
        data_chunk.columns = ['Glycan', 'Lectin']
        f_column = rescale_rfu(rfu_column)

        concentration = float(subset['Amount'].values[0].split(' ')[0])
        data_chunk['Concentration'] = concentration
        data_chunk['Fraction_Bound'] = f_column
        data_chunk = data_chunk.reset_index(drop = True)
        #print(data_chunk)

        f_data = pandas.concat((f_data, data_chunk), axis = 'index')

        if subset['Lectin'].values[0] not in zeros:
            all_glycans = rfu_data['GlycanID'].drop_duplicates()
            data_chunk = pandas.DataFrame({'Glycan': all_glycans})
            data_chunk['Lectin'] = subset['Lectin'].values[0]
            data_chunk['Concentration'] = 0.0
            data_chunk['Fraction_Bound'] = 0.0
            f_data = pandas.concat((f_data, data_chunk), axis = 'index')
            zeros = zeros.union(set([subset['Lectin'].values[0]]))
            
    return f_data



# read in CFG data and get RFUs
cfg = read_CFG_table('SendToClass/CFG-v5-Table.txt.gz')
cfg = cfg.reset_index().set_index('ObjId')

regex = re.compile(r'CFG-0*')
cfg['GlycanID'] = [regex.sub('CFG-', x)  for x in cfg['GlycanID']]

# read in the samples table
samples = pandas.read_csv('CFG-Lectins-Reduction-final2.tsv', sep = '\t',
                          index_col = 'objId', float_precision = 'round_trip')

# eliminate rows without a sequence or that are marked bad
mask = (samples['Sequence'] != "") & (samples['X'] != 'A')
samples = samples[mask]
samples['Sequence'] = [x.upper()  for x in samples['Sequence']]

# find the unique sequences and assign each a number 1, 2, 3, ...
sequences = samples[['Sequence']].drop_duplicates()
sequences = sequences.drop_duplicates().reset_index(drop = True)
sequences['Lectin'] = sequences.index + 1
sequences = sequences.set_index('Sequence')

# merge the Lectin Numbers with the sample table
s2 = samples.reset_index().set_index('Sequence').join(sequences)
s2 = s2[['objId', 'Amount', 'Lectin']].set_index('objId')

cfg = cfg.join(s2, how = 'right')
f_bound = convert_to_f(cfg)

folds = pandas.read_csv('running/Data/Folds_10_CFG611.tsv.gz',
                        sep = '\t', index_col = 'Name')
folds['Name'] = [regex.sub('CFG-', x)  for x in folds.index]
folds = folds.set_index('Name')

f_bound = f_bound.set_index('Glycan').join(folds)

f_bound['Fraction_Bound'] = f_bound['Fraction_Bound'].map(lambda x: '{:0.9f}'.format(x))
f_bound.to_csv('Fraction_Bound.txt.gz', sep = '\t')
sequences.reset_index().set_index('Lectin').to_csv('Lectin-Sequences.txt', sep ='\t')
print(f_bound)

sys.exit(1)


# read in the samples table
### samples = pandas.read_csv('sample-table-rev.tsv', sep = '\t',
###                           index_col = 'ObjId', float_precision = 'round_trip')




### kde = sklearn.neighbors.KernelDensity(kernel="gaussian", bandwidth=0.75).fit(X)
### log_dens = kde.score_samples(X_plot)
### ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc="#AAAAFF")



cfg =  convert_RFU_to_f(cfg, samples)
# get rid of some unneed columns
cfg = cfg.drop(['SampleNum', 'Base', 'Maximum', 'Description-1', 'RFU'], axis = 1)

cols = cfg.columns.tolist()
#cols[0] = 'ProteinGroup'
cfg.columns = cols
print(cols)
cfg.to_csv('testfile-name', sep = '\t')



#sys.exit(1)

# build a list of concentrations
conc_list = [float('{}e{}'.format(i, j-1))  for j in [-5, -4, -3, -2, -1, 0, 1]
                                 for i in [10, 15, 20, 30, 40, 50, 70]] + [0.0, 100.0]

conc_frame = pandas.DataFrame(math.nan, index = conc_list,
                               columns = ['f'])
conc_frame.loc[0.0] = 0.0


outputs = []
#for groupnum in [6, 82, 83]:
for groupnum in cfg['GroupNum'].unique():

    subset = cfg.set_index('GroupNum').loc[groupnum]
    
    print('Processing Group:', groupnum, subset['Concentration'].unique())
    subset = subset.set_index(['GlycanID', 'Concentration'], append = True)
    for glycan in subset.index.get_level_values(level = 'GlycanID').unique():
        series = subset.xs(glycan, level = 'GlycanID')

        x = interpolate_series(series, conc_frame)
        
        # add some labels
        x['GroupNum'] = groupnum
        x['GlycanID'] = glycan
        outputs.append(x)
        
# convert the results into one dataframe
outputs = pandas.concat(outputs)
outputs.index = outputs.index.set_names('Concentration')

outputs = outputs.set_index('GlycanID', append = True)

outputs.to_csv('interpolated-Fraction-Bound-Table-prepivot.txt.gz', sep = '\t')

outputs

# pivot from long format to wide - one column for each sample group        
outputs = outputs.pivot(columns = 'GroupNum', values = 'f')
outputs = outputs.fillna(-1)
# save the results
outputs.to_csv('interpolated-Fraction-Bound-Table-NoInterp.txt.gz', sep = '\t')
