#!/usr/bin/env python
'''example: ./sas2csv.py "*.sas7bdat"'''

from glob import glob
import sas7bdat
import sys
import traceback

def convert(path):
    files = glob(path)
    print 'Converting %d files in %s' % (len(files), path)
    for f in files:
        if f[-9:] == '.sas7bdat':
            fout = f[:-9] + '.csv'
            try:
                raw = sas7bdat.SAS7BDAT(f)
                raw.convert_file(fout)
            except:
                traceback.print_last()

if __name__ == '__main__':
    convert(sys.argv[1])    