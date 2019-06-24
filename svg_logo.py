import numpy as np
import pandas as pd
import svgwrite
from svgwrite import cm, mm
# import matplotlib.pyplot as plt

HW_RATIO = 0.5

__all__ = ['svg_logo']

def oneAA(dwg, aa, ul, lr, fontsize=2, units=cm, color='black'):
    fontwidth  = fontsize * HW_RATIO * 1.2
    fontheight = fontsize * 0.7

    width  = lr[0] - ul[0]
    height = lr[1] - ul[1]

    x_scale = width / fontwidth
    y_scale = height / fontheight
    tx = 'scale(%s, %s)' % (x_scale, y_scale)
    ll = (ul[0]/x_scale, lr[1]/y_scale)
    #print(aa, tx)
    #box = dwg.add(dwg.rect(insert=(ll[0]*units, ll[1]*units), size=(0.1*units, 0.1*units), fill='orange', transform=tx))
    txt = dwg.add(dwg.text(aa, x=[(ll[0]) * units],
                            y=[(ll[1]) * units],
                            transform=tx,
                            fill=color,
                            font_size='%dcm' % fontsize,
                            font_family='Lucida Console',
                            font_weight='normal'))
    return dwg

def svg_logo(motif, filename, aa_colors='black', highlightAAs=None, highlightColor='red', convert=False):
    """Sequence logo of the motif using SVG.

    Parameters
    ----------
    motif : pd.DataFrame
        Matrix of scores for each AA symbol (index, rows) and each
        column in an alignment (columns). Values will be used to scale
        each symbol linearly.
    aa_colors : str
        Either 'shapely' or a color to use for all symbols.
    highlightAAs : list of lists
        Lists AAs in each position that should be plotted in highlight color.
    highlightColor : string or RGB
        Valid color for highlighted AAs"""
    fontsize = 2
    X = fontsize * HW_RATIO
    H,W = 30, motif.shape[0] * X / 2
    Z = H/2
    
    dwg = svgwrite.Drawing(filename=filename, height='100%', width='100%')#, viewBox='0 0 100 100')# % int(a.shape[1]*1))

    if highlightAAs is None:
        highlightAAs = ['' for i in range(motif.shape[1])]
    if aa_colors == 'shapely':
        colorsDf = pd.read_csv(io.StringIO(_shapely_colors), delimiter=',')
        aa_colors = {aa:parseColor(aa, colorsDf=colorsDf) for aa in motif.index}
    else:
        """All AAs have the same color, specified by aa_colors"""
        aa_colors = {aa:aa_colors for aa in motif.index}
    #bbox = dwg.add(dwg.rect(insert=(0, 0), size=(W*cm, H*cm), stroke='orange', fill='white'))
    #zero = dwg.add(dwg.line(start=(0, Z*cm), end=(W*cm, Z*cm), stroke='black'))
    for xi in range(motif.shape[1]):
        scores = motif.iloc[:, xi]
        pos_scores = scores[scores > 0].sort_values()
        neg_scores = (-scores[scores < 0]).sort_values()
        
        posshift = 0
        for yi, (aa, score) in enumerate(pos_scores.items()):
            if aa in highlightAAs[xi]:
                color = highlightColor
            else:
                color = aa_colors[aa]
            x = (xi*X)
            y = (Z - posshift - score*fontsize)
            
            #box = dwg.add(dwg.rect(insert=(x*cm, y*cm), size=(X*cm, score*fontsize*cm), fill='green', opacity=score))
            txt = oneAA(dwg, aa, ul=(x, y), lr=(x+X, y+score*fontsize), color='black')
            
            #print(aa, x, y, score)
            posshift += score*fontsize

        negshift = 0
        for yi, (aa, score) in enumerate(neg_scores.items()):
            if aa in highlightAAs[xi]:
                color = highlightColor
            else:
                color = aa_colors[aa]
            x = (xi*X)
            y = (Z + negshift)

            #box = dwg.add(dwg.rect(insert=(x*cm, y*cm), size=(X*cm, score*fontsize*cm), stroke='red', opacity=score))
            txt = oneAA(dwg, aa, ul=(x, y), lr=(x+X, y+score*fontsize), color='red')
            
            #print(aa, x, y, score)
            negshift += score*fontsize
    dwg.save()

    if convert:
        import subprocess

        cmd = ['convert', '-density 200', filename, filename.replace('.svg', '') + '.png']
        subprocess.call(' '.join(cmd), shell=True)
        filename = filename.replace('.svg', '') + '.png'
    return filename

if __name__ == '__main__':
    a = pd.DataFrame(np.array([[0.1, -0.2, 0, 0.5],[0, 0.3, -0.8, 0.1]]).T, columns=[0, 1], index=['C', 'R', 'W', 'I'])
    print(a)
    svg_logo(a, 'test.svg')
    import subprocess

    cmd = ['convert', '-density 200', 'test.svg', 'test.png']
    subprocess.call(' '.join(cmd), shell=True)

    
