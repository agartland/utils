import time
import subprocess
import os.path as op
import pandas as pd

__all__ = ['toPNG', 'toPDF']

def toPNG(df, outFn, titStr, float_format='%1.3g', dpi=200):
    assert outFn[-4:] == '.png'
    folder,fn = op.split(outFn)
    pdfFn = outFn.replace('.png', '.pdf')
    toPDF(df, pdfFn, titStr, float_format=float_format)
    cmd = ['convert', '-density %d' % dpi, pdfFn, outFn]
    #print ' '.join(cmd)
    subprocess.check_call(' '.join(cmd), shell=True)

def toPDF(df, outFn, titStr, float_format='%1.3g'):
    folder,fn = op.split(outFn)
    df = df.applymap(lambda s: s if not isinstance(s, basestring) else s.replace('_','-'))
    df = df.rename_axis(lambda s: s.replace('_','-'), axis=1)
    texFn = outFn[:-3] + 'tex'
    header = ['\\documentclass[10pt]{article}',
              '\\usepackage{lmodern}',
              '\\usepackage{booktabs}',
              '\\usepackage{longtable}',
              '\\usepackage{geometry}',
              '\\usepackage[english]{babel}',
              '\\usepackage[utf8]{inputenc}',
              '\\usepackage{fancyhdr}',
              '\\geometry{letterpaper, landscape, margin=1in}',
              '\\pagestyle{fancy}',
              '\\fancyhf{}',
              '\\rhead{%s}' % time.ctime(),
              '\\chead{%s}' % titStr,
              '\\rfoot{Page \\thepage}',
              '\\begin{document}']
    
    footer = ['\\end{document}']
    with open(texFn,'w') as fh:
        for h in header:
            fh.write(h + '\n')
        fh.write(df.to_latex(float_format=lambda f: float_format % f,
                             longtable=True, index=False, escape=False))
        for f in footer:
            fh.write(f + '\n')
    cmd = ['latex', '-output-format=pdf', '-output-directory=%s' % folder, texFn]
    subprocess.call(cmd)
    """Run latex twice to get the layout correct"""
    subprocess.call(cmd)