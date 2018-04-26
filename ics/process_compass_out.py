import sys
import pandas as pd

"""Reformats the CSV output from a COMPASS run as a human navigable CSV"""

fn = sys.argv[1]

df = pd.read_csv(fn)

"""Parse the marker columns as 0/1 indicators"""
markers = df.columns[1].replace('!', '').split('&')

cytColumns = df.columns[1:]
cytTuples = []
for col in cytColumns:
    t = tuple([0 if '!' == c[0] else 1 for c in col.split('&')])
    cytTuples.append(t)

"""Parse the sample info and use as an index"""
tmp = df.iloc[:, 0].str.split(':')
df.loc[:, 'Antigen'] = tmp.map(lambda l: l[0].replace('"', ''))
df.loc[:, 'PTID'] = tmp.map(lambda l: l[1])
df.loc[:, 'Visit'] = tmp.map(lambda l: l[2])


df = df.set_index(['PTID', 'Visit', 'Antigen'])[cytColumns]
df.columns = pd.MultiIndex.from_tuples(cytTuples, names=markers)

"""Stack hierarchical/combinatorial columns as rows"""
for cy in markers:
    df = df.stack(0)

df.name = 'Pr(resp)'

outFn = fn[:-4] + '_formatted' + '.csv'
df.reset_index(level=markers).to_csv(outFn, index=True)