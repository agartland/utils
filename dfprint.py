import time
import subprocess
import os.path as op
import pandas as pd
import os
from functools import partial

__all__ = ['toPNG', 'toPDF']

def toPNG(df, outFn, dpi=200, **kwargs):
    assert outFn[-4:] == '.png'
    folder, fn = op.split(outFn)
    pdfFn = outFn.replace('.png', '.pdf')
    toPDF(df, pdfFn, **kwargs)
    cmd = ['convert',# '-interaction=nonstopmode',
           '-density %d' % dpi,
           pdfFn,
           outFn]
    #print ' '.join(cmd)
    if 'hideConsole' in kwargs and kwargs['hideConsole']:
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            #si.wShowWindow = subprocess.SW_HIDE # default
        except:
            si = None
    else:
        si = None

    devnull = open(os.devnull, 'w')
    subprocess.check_call(' '.join(cmd),
                          shell=True,
                          startupinfo=si,
                          stdout=devnull,
                          stderr=devnull)
    devnull.close()
    removeAuxFiles(outFn)

def toPDF(df,
          outFn,
          titStr='',
          float_format='%1.3g',
          index=False,
          hideConsole=True,
          landscape=True,
          legal=False,
          margin=1):
    if landscape:
        orientation = 'landscape'
    else:
        orientation = 'portrait'

    if not legal:
        paper = 'letterpaper'
    else:
        paper = 'legalpaper'

    folder, fn = op.split(outFn)
    
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    def repChar(s, c1, c2):
        if not isinstance(s, str):
            return s
        else:
            return s.replace(c1, c2)
    if not df.empty:
        for func in [partial(repChar, c1='_', c2='-'),
                     partial(repChar, c1='%', c2='PCT')]:
            df = df.applymap(func)
            df = df.rename_axis(func, axis=0)
            df = df.rename_axis(func, axis=1)

    texFn = outFn[:-3] + 'tex'
    header = ['\\documentclass[10pt]{article}',
              '\\usepackage{lmodern}',
              '\\usepackage{booktabs}',
              '\\usepackage{longtable}',
              '\\usepackage{geometry}',
              '\\usepackage[english]{babel}',
              '\\usepackage[utf8]{inputenc}',
              '\\usepackage{fancyhdr}',
              '\\geometry{%s, %s, margin=%1.1fin}' % (paper, orientation, margin),
              '\\pagestyle{fancy}',
              '\\fancyhf{}',
              '\\rhead{%s}' % time.ctime(),
              '\\chead{%s}' % titStr,
              '\\rfoot{Page \\thepage}',
              '\\begin{document}']
    
    footer = ['\\end{document}']

    with open(texFn, 'w') as fh:
        for h in header:
            fh.write(h + '\n')
        sout = df.to_latex(float_format=lambda f: float_format % f,
                             longtable=True, index=index, escape=False)
        fh.write(sout)
        for f in footer:
            fh.write(f + '\n')
    cmd = ['latex',
           '-output-format=pdf',
           '-output-directory=%s' % folder,
           texFn]
    
    if hideConsole:
        try:
            """This won't work in linux"""
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            # si.wShowWindow = subprocess.SW_HIDE # default
        except:
            si = None
        cmd.insert(2, '-interaction=nonstopmode')
    else:
        si = None
    

    devnull = open(os.devnull, 'w')
    for i in range(2):
        """Run latex twice to get the layout correct"""
        subprocess.call(cmd,
                        startupinfo=si,
                        stdout=devnull,
                        stderr=devnull)
    devnull.close()
    removeAuxFiles(outFn)

def removeAuxFiles(outFn):
    extensions = ['aux', 'log', 'tex']
    for ext in extensions:
        fn = outFn[:-3] + ext
        try:
            os.remove(fn)
        except OSError:
            pass
