import numpy as np
import pandas as pd
import svgwrite
from svgwrite import cm, mm
# import io
# import matplotlib.pyplot as plt
from svg_alphabet import add_letter
import matplotlib as mpl

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



def svg_logo(motif, filename, aa_colors='black', highlightAAs=None, highlightColor='red', convert=False, return_str=False):
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
    if not return_str:
        dwg.save()

        if convert:
            import subprocess

            cmd = ['convert', '-density 200', filename, filename.replace('.svg', '') + '.png']
            subprocess.call(' '.join(cmd), shell=True)
            filename = filename.replace('.svg', '') + '.png'
        return filename
    else:
        return str(dwg)

"""TODO
 - Use letters from svg_alphabet (each is 100 x 100)
 - Heights are just a scaling relative to 1.
 - Do positives first, then negatives.
 - Add them one at a time to a drawing with translation and scaling attributes
 - Add axes and axes ticks with labels.
 - Set viewBox to extent of columns and max height
"""

def new_svg_logo(motif, filename):
    margin = 10
    left_margin = 50
    bottom_margin = 50
    xpad = 4
    HW = 100

    """Scale height of 100px to absolute max value"""
    mx_value = np.max(np.abs(motif.values))
    hscale = HW / mx_value
    wscale = 1

    mx_pos = 0
    mn_neg = 0
    for j in range(motif.shape[1]):
        tmp = motif.values[:, j]
        tot = np.sum(tmp[tmp>0])
        if tot > mx_pos:
            mx_pos = tot
        tot = np.sum(tmp[tmp<0])
        if tot < mn_neg:
            mn_neg = tot
    
    yticklabels = mpl.ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(mn_neg, mx_pos)
    yticks = [hscale * yt for yt in yticklabels]

    mx_pos = hscale * mx_pos
    mn_neg = hscale * mn_neg

    mx_pos = np.max([mx_pos, yticks[-1]])
    mn_neg = np.min([mn_neg, yticks[0]])

    yzero = margin + mx_pos
    xzero = left_margin

    height = mx_pos + mn_neg + margin + bottom_margin
    width = wscale * (HW + xpad) * motif.shape[1] + margin + left_margin + xpad

    xticks = [xzero + (i + 1) * wscale * (HW + xpad) - wscale * HW / 2 for i in range(motif.shape[1])]
    xticklabels = ['%d' % (i + 1) for i in range(motif.shape[1])]

    dwg = svgwrite.Drawing(filename=filename, height=height, width=width)#, viewBox='0 0 100 100')# % int(a.shape[1]*1))

    letter_groups = {}
    for xi in range(motif.shape[1]):
        xshift = xzero + xi*(HW + xpad) + xpad
        scores = motif.iloc[:, xi]
        pos_scores = scores[scores > 0].sort_values()
        neg_scores = (-scores[scores < 0]).sort_values()
        
        posshift = 5
        for yi, (aa, score) in enumerate(pos_scores.items()):
            scaled_height = hscale * score
            translate = (xshift, yzero - posshift - scaled_height)
            transform = 'translate({xtrans} {ytrans}) scale({xscale} {yscale})'.format(xtrans=translate[0], ytrans=translate[1],
                                                                                        xscale=wscale, yscale=score/mx_value)
            letter_groups[(xi, aa)] = add_letter(dwg, aa, group_id='%d_%s' % (xi, aa), color=None, background='white', transform=transform)
            #box = dwg.add(dwg.rect(insert=(x*cm, y*cm), size=(X*cm, score*fontsize*cm), fill='green', opacity=score))
            #print(aa, x, y, score)
            posshift += scaled_height

        negshift = 5
        for yi, (aa, score) in enumerate(neg_scores.items()):
            scaled_height = hscale * score
            translate = (xshift, yzero + negshift)
            transform = 'translate({xtrans} {ytrans}) scale({xscale} {yscale})'.format(xtrans=translate[0], ytrans=translate[1],
                                                                                        xscale=wscale, yscale=score/mx_value)
            letter_groups[(xi, aa)] = add_letter(dwg, aa, group_id='%d_%s' % (xi, aa), color=None, background='white', transform=transform)
            #box = dwg.add(dwg.rect(insert=(x*cm, y*cm), size=(X*cm, score*fontsize*cm), fill='green', opacity=score))
            #print(aa, x, y, score)
            negshift += scaled_height
    
    axes = dwg.add(dwg.g(id='axes', stroke_width=1.5, stroke='#000000'))
    axes.add(dwg.path(d='M {xz} {yz} h {len}'.format(xz=xzero, yz=yzero, len=wscale * (HW + xpad) * motif.shape[1] + 2 * xpad)))
    axes.add(dwg.path(d='M {xz} {yz} v {len}'.format(xz=xzero, yz=yzero - np.min(yticks), len=-(yticks[-1] - yticks[0]))))
    for yt, ytl in zip(yticks, yticklabels):
        axes.add(dwg.path(d='M {xz} {yz} h {len}'.format(xz=xzero, yz=yzero - yt, len=-(margin / 2))))
        axes.add(dwg.text(ytl, (xzero - (margin/2) - xpad, yzero - yt),
                          fill='#000000',
                          font_size='10pt',
                          font_family='sans-serif',
                          font_weight='normal',
                          text_anchor='end',
                          dominant_baseline='middle'))
    for xt, xtl in zip(xticks, xticklabels):
        axes.add(dwg.text(xtl, (xt, yzero - yticks[0] + 10),
                          fill='#000000',
                          font_size='10pt',
                          font_family='sans-serif',
                          font_weight='normal',
                          text_anchor='middle',
                          dominant_baseline='hanging'))

    dwg.save()


if __name__ == '__main__':
    # a = pd.DataFrame(np.array([[0.1, -0.2, 0, 0.5],[0, 0.3, -0.8, 0.1]]).T, columns=[0, 1], index=['C', 'R', 'W', 'S'])
    # a = pd.DataFrame(np.array([[0.1, 0.2, 0, 0.5],[0, 0.3, 0.8, 0.1], [0.5, 0.1, 0.4, 0.]]).T, columns=[0, 1, 2], index=['C', 'R', 'W', 'S'])
    a = pd.DataFrame(np.array([[0.1, 0.2, 0, 0.5],[0, 0.3, 0.8, 0.1], [0.5, 0.1, 0.4, 0.]]).T, columns=[0, 1, 2], index=['C', 'R', 'W', 'S'])
    new_svg_logo(a, 'test.svg')
    '''
    import subprocess

    cmd = ['convert', '-density 200', 'test.svg', 'test.png']
    subprocess.call(' '.join(cmd), shell=True)
    '''

    
