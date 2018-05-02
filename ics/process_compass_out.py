import sys
import pandas as pd

"""Reformats the CSV output from a COMPASS run as a human navigable CSV"""

fn = sys.argv[1]
# fn = r'T:/vaccine/p602/analysis/lab/pt_reports/ics/2018_02_compass/adata/hvtn602_compass_results'

prFn = fn + '_mean_gamma.csv'
stimFn = fn + '_stim_counts.csv'
unstimFn = fn + '_unstim_counts.csv'

def processFile(fn):
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
    return df

prS = processFile(prFn)
prS.name = 'Pr(resp)'

stimS = processFile(stimFn)
stimS.name = 'Stim positive'

stimDf = stimS.reset_index().set_index(['PTID', 'Visit', 'Antigen'])
ssCols = [c for c in stimDf.columns if not c == 'Stim positive']

stimTot = stimS.groupby(level=['PTID', 'Visit', 'Antigen']).agg(np.sum)
stimTot.name = 'Stim total'

stimDf = stimDf.join(stimTot).reset_index().set_index(['PTID', 'Visit', 'Antigen'] + ssCols)

unstimS = processFile(unstimFn)
unstimS.name = 'Unstim positive'
unstimDf = unstimS.reset_index().set_index(['PTID', 'Visit', 'Antigen'])

unstimTot = unstimS.groupby(level=['PTID', 'Visit', 'Antigen']).agg(np.sum)
unstimTot.name = 'Unstim total'

unstimDf = unstimDf.join(unstimTot).reset_index().set_index(['PTID', 'Visit', 'Antigen'] + ssCols)

outDf = pd.concat((prS, stimDf, unstimDf), axis=1)

outDf.loc[:, 'Stim pctpos'] = 100 * outDf['Stim positive'] / outDf['Stim total']
outDf.loc[:, 'Unstim pctpos'] = 100 * outDf['Unstim positive'] / outDf['Unstim total']
outDf.loc[:, 'Pctpos adj'] = outDf['Stim pctpos'] - outDf['Unstim pctpos']

outFn = fn + '_formatted' + '.csv'
outDf.reset_index(level=ssCols).to_csv(outFn, index=True)