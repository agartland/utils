"""Generate HVTN505 dataset for Michael on statsrv"""
import pandas as pd
import numpy as np

"""Read in the raw ICS data"""
fn = '/trials/vaccine/p505/analysis/lab/pdata/ics/e505ics_fh/csvfiles/e505ics_fh_p.csv'
ctrlCols = ['ptid', 'visitday', 'tcellsub', 'cytokine']
indexCols = ctrlCols + ['antigen']
uAg = ['CMV',
       'Empty Ad5 (VRC)',
       'VRC ENV A',
       'VRC ENV B',
       'VRC ENV C',
       'VRC GAG B',
       'VRC NEF B',
       'VRC POL 1 B',
       'VRC POL 2 B']

rdf = pd.read_csv(fn, usecols=indexCols + ['nsub', 'cytnum', 'nrepl'],
                      dtype={'ptid':object,
                             'visitday':np.int,
                             'tcellsub':object,
                             'cytokine':object,
                             'antigen':object,
                             'nsub':np.int,
                             'cytnum':np.int,
                             'nrepl':np.int},
                      index_col=indexCols).sort_index()


"""Sum the negative control replicates"""
ndf = rdf.xs('negctrl', level='antigen').reset_index().groupby(ctrlCols)[['nsub', 'cytnum']].agg(np.sum)
ndf.loc[:, 'bg'] = ndf['cytnum'] / ndf['nsub']

"""Define the magnitude as the fraction of cytokine positive cells"""
pdf = rdf.loc[(slice(None), slice(None), slice(None), slice(None), uAg), :]
pdf.loc[:, 'mag'] = pdf['cytnum'] / pdf['nsub']

"""Subtract off the background/negative control"""
df = pdf['mag'].reset_index().join(ndf['bg'], on=ctrlCols)

"""Create a wide dataset with hierarchical columns"""
wideDf = df.set_index(indexCols).unstack(['visitday', 'tcellsub', 'antigen', 'cytokine'])

"""Flatten the columns as a string with "|" separator"""
def strcatFun(iter, sep='|'):
    s = ''
    for v in iter:
        s += str(v) + sep
    return s[:-len(sep)]
    
strCols = [strcatFun(c) for c in wideDf.columns.tolist()]
outDf = wideDf.copy()
outDf.columns = strCols
outDf.to_csv('hvtn505_ics_24Jul2017.csv')

"""Save column metadata"""
colMeta = wideDf.columns.to_frame().drop(0, axis=1)
colMeta.index = strCols
colMeta.index.name = 'column'
colMeta.to_csv('hvtn505_ics_colmeta_24Jul2017.csv')


"""Load PTID rx data"""
rxFn = '/trials/vaccine/p505/analysis/adata/rx_v2.csv'
trtCols = ['ptid', 'arm', 'grp', 'rx_code', 'rx', 'pub_id']
tmp = pd.read_csv(rxFn)
tmp = tmp.rename_axis({'Ptid': 'ptid'}, axis=1)
tmp.loc[:, 'ptid'] = tmp.ptid.str.replace('-', '')
trtDf = tmp[trtCols].set_index('ptid')
trtDf.to_csv('hvtn505_ics_rowmeta_24Jul2017.csv')