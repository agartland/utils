#!/usr/bin/env python

"""
Usage examples:
python /home/agartlan/gitrepo/utils/ics/mergeFlowData.py --base two-of-three

sbatch -n 1 -t 3-0 -c 4 -o functions_slurm.txt --wrap="python /home/agartlan/gitrepo/utils/ics/mergeGatingSets.py --function functions --ncpus 4 --out functions_extract.csv"

To delete all tmp files use:
find . -name \merged_tmp*.feather -type f -delete
"""
def _passthrough(df):
    df.columns = [c.split('.')[-1] for c in df.columns]
    return df

def mergeBatches(dataFolder, outFile, testsamples=False, testbatch=False, matchStr='two-of-three*.feather', metaCols=None, filters=None):
    out = []
    batchList = [opj(dataFolder, bf) for bf in os.listdir(dataFolder) if os.path.isdir(opj(dataFolder, bf))]
    if testbatch:
        batchList = batchList[:1]

    func = partial(mergeSamples,
                   extractionFunc=_passthrough,
                   extractionKwargs={},
                   matchStr=matchStr,
                   test=testsamples,
                   metaCols=metaCols,
                   filters=filters)
    res = list(map(func, batchList))

    outFilename = mergeFeathers(res, outFile, writeCSV=False)
    return outFilename

def testMatching(dataFolder):
    out = []
    for bf in os.listdir(dataFolder):
        batchFolder = opj(dataFolder, bf)
        if os.path.isdir(opj(dataFolder, bf)):
            featherLU = matchSamples(batchFolder, test=False)
            tmp = pd.Series(featherLU).to_frame()
            tmp.loc[:, 'batch'] = bf
            tmp.loc[:, 'batch_folder'] = opj(dataFolder, bf)
            out.append(tmp)
    return pd.concat(out, axis=0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Combine files containing single cell readouts.')
    parser.add_argument('--folder', type=str,
                        help='Data folder containing all batch folders.',
                        default='/fh/fast/gilbert_p/grp/compass_hvtn602_aw/tmpdata')
    parser.add_argument('--base', type=str,
                        help='Base of the file name to search for',
                        default='two-of-three')
    parser.add_argument('--testsamples', action='store_true', help='Only process two samples from each batch.')
    parser.add_argument('--testbatch', action='store_true', help='Only process twp samples from one batch.')
    parser.add_argument('--matchingonly', action='store_true', help='Only perform sample matching, to validate metadata.')
    parser.add_argument('--utils', default='/home/agartlan/gitrepo/utils', help='Location of agartland/utils repo from public github.com')
    
    args = parser.parse_args()

    import itertools
    import pandas as pd
    import numpy as np
    from os.path import join as opj
    import os
    from functools import partial
    import time
    import sys
    import feather
    
    """Make sure the utils are on path before importing"""
    sys.path.append(args.utils)

    # from ics import extractFunctionsGBY, extractFunctionsMarkersGBY, parseSubsets, mergeSamples, matchSamples
    from ics import *

    if args.matchingonly:
        metaDf = testMatching(args.folder)
        metaDf.to_csv(opj(args.folder, 'metamatch_' + args.out))
        print('Wrote matching metadata to %s.' % opj(args.folder, 'metamatch_' + args.out))
    else:
        if args.testbatch:
            print('Test: processing samples from one batch')

        if args.testsamples:
            print('Test: processing two samples per batch')
        
        outFile = opj(args.folder, args.base + '_combined.feather')
        
        wrote = mergeBatches(args.folder,
                          testsamples=args.testsamples,
                          testbatch=args.testbatch,
                          outFile=outFile,
                          matchStr=args.base + '*.feather',
                          metaCols=['PTID', 'VISITNO', 'Global.Spec.Id', 'TESTDT', 'STIM'],
                          filters={'STIM':['negctrl', 'TB WCL', 'BCG-Pasteur', 'Ag85B', 'TB 10.4'], 'VISITNO':[2, 10]})

        if wrote == outFile:
            print('Wrote extracted data to %s.' % outFile)
        else:
            print('Error writing file to disk: %s' % wrote)