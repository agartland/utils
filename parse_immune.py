import pandas as pd
import numpy as np

__all__ = ['parseICS',
           'parseBAMA',
           'parseNAB',
           'unstackIR']

def unstackIR(df, uVars):
    """Return a df with one row per ptid
    and columns for each combination of uVars"""
    pass

def parseICS(fn, uVars = ['visitno','tcellsub','cytokine','antigen'], subset={}, mag='pctpos_adj'):
    """Parse a processed ICS file.
    Returns one row per response, subsetting on subset values."""

    raw = pd.read_csv(fn, dtype = {'ptid':str}, skipinitialspace = True)
    
    cols = ['ptid','response','mag'] + uVars
    raw['mag'] = raw[mag]
    
    return raw[cols]
def parseBAMA():
    pass
def parseNAB():
    pass

"""Existing code from mubiomeCOR"""
def pullData(fn, visitno, dataType, fullDf, subset=None, protid=None, day=None):
    def lookupCode(ptid,field):
        code = fullDf[field].loc[ptid]
        if not type(code) is str:
            code = code.iloc[0]
        return code
    tmp = pd.read_csv(fn, dtype = {'ptid':str}, skipinitialspace = True)
    if dataType == 'BAMA':
        #cols = ['protocol','ptid','antigen','response','delta','rx_code','antigen_label','visitno']
        tmp['assay'] = 'BAMA %s' % (day)
        tmp['mag'] = tmp.delta
        tmp['protid'] = tmp.protocol.map(lambda p: '%03.0f' % p)
    elif dataType == 'ICS':
        tmp['assay'] = tmp.tcellsub
        tmp['mag'] = tmp.pctpos_adj
        tmp['protid'] = protid
    """TODO:
    elif dataType == 'NAB':
        tmp['assay'] = 
        tmp['mag'] = 
        tmp['protid'] = 
    """
    tmp['ptid'] = tmp.ptid.str.replace('-','')
    #print sum(map(lambda ptid: ptid in subset,tmp.ptid.unique()))
    if not subset is None:
        tmp = tmp.loc[(tmp.visitno==visitno) & (tmp.ptid.map(lambda ptid: ptid in subset))]
    else:
        tmp = tmp.loc[tmp.visitno==visitno]

    if not 'rx_code' in tmp.columns:
        tmp['rx_code'] = tmp.ptid.map(partial(lookupCode,field = 'rx_code'))
    tmp['site'] = tmp.ptid.map(partial(lookupCode,field = 'Site'))
    tmp['sex'] = tmp.ptid.map(partial(lookupCode,field = 'DEMsex'))
    return tmp.drop_duplicates()

def assembleIR(tmp,nameInd=None):
    """Put all BAMA antigens in a single df"""
    allIR = None
    gb = tmp[['assay','antigen','mag','ptid']].groupby(['assay','antigen'])
    keys = gb.groups.keys()
    for k in keys:
        oneCol = gb.get_group(k)[['mag','ptid']].set_index('ptid').copy()
        if nameInd is None:
            oneCol = oneCol.rename_axis({'mag':'-'.join(k)},axis=1)
        else:
            oneCol = oneCol.rename_axis({'mag':k[nameInd]},axis=1)
        if allIR is None:
            allIR = oneCol
        else:
            allIR = allIR.join(oneCol, how = 'outer')
    return allIR