import time
import subprocess
import os.path as op
import pandas as pd

__all__ = ['toPNG', 'toPDF']

def toPNG(df, outFn, dpi=200, **kwargs):
    assert outFn[-4:] == '.png'
    folder,fn = op.split(outFn)
    pdfFn = outFn.replace('.png', '.pdf')
    toPDF(df, pdfFn, **kwargs)
    cmd = ['convert',
           '-interaction=nonstopmode',
           '-density %d' % dpi,
           pdfFn,
           outFn]
    #print ' '.join(cmd)
    
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    #si.wShowWindow = subprocess.SW_HIDE # default

    subprocess.check_call(' '.join(cmd), shell=True, startupinfo=si)

def toPDF(df, outFn, titStr='', float_format='%1.3g', index=False, hideConsole=True):
    repUnderscore = lambda s: s if not isinstance(s, basestring) else s.replace('_','-')

    folder,fn = op.split(outFn)
    df = df.applymap(repUnderscore)
    df = df.rename_axis(repUnderscore, axis=1)
    df = df.rename_axis(repUnderscore, axis=0)
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
                             longtable=True, index=index, escape=False))
        for f in footer:
            fh.write(f + '\n')
    cmd = ['latex',
           '-output-format=pdf',
           '-output-directory=%s' % folder,
           texFn]
    
    if hideConsole:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        # si.wShowWindow = subprocess.SW_HIDE # default
        cmd.insert(2,'-interaction=nonstopmode')
    else:
        si = None
    
    subprocess.call(cmd, startupinfo=si)
    """Run latex twice to get the layout correct"""
    subprocess.call(cmd, startupinfo=si)
