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

    subprocess.check_call(' '.join(cmd), shell=True, startupinfo=si)

def toPDF(df, outFn, titStr=u'', float_format=u'%1.3g', index=False, hideConsole=True):
    repUnderscore = lambda s: s if not isinstance(s, basestring) else s.replace('_','-')

    folder,fn = op.split(outFn)
    df = df.applymap(repUnderscore)
    df = df.rename_axis(repUnderscore, axis=1)
    df = df.rename_axis(repUnderscore, axis=0)
    texFn = outFn[:-3] + 'tex'
    header = [u'\\documentclass[10pt]{article}',
              u'\\usepackage{lmodern}',
              u'\\usepackage{booktabs}',
              u'\\usepackage{longtable}',
              u'\\usepackage{geometry}',
              u'\\usepackage[english]{babel}',
              u'\\usepackage[utf8]{inputenc}',
              u'\\usepackage{fancyhdr}',
              u'\\geometry{letterpaper, landscape, margin=1in}',
              u'\\pagestyle{fancy}',
              u'\\fancyhf{}',
              u'\\rhead{%s}' % time.ctime(),
              u'\\chead{%s}' % titStr,
              u'\\rfoot{Page \\thepage}',
              u'\\begin{document}']
    
    footer = [u'\\end{document}']

    with open(texFn,'w') as fh:
        for h in header:
            fh.write(h + u'\n')
        sout = df.to_latex(float_format=lambda f: float_format % f,
                             longtable=True, index=index, escape=False)
        fh.write(sout)
        for f in footer:
            fh.write(f + u'\n')
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
        cmd.insert(2,'-interaction=nonstopmode')
    else:
        si = None
    
    subprocess.call(cmd, startupinfo=si)
    """Run latex twice to get the layout correct"""
    subprocess.call(cmd, startupinfo=si)
