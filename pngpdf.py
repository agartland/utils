import os
from os.path import join as opj

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib._pylab_helpers import Gcf

__all__ = ['PngPdfPages']

class PngPdfPages(PdfPages):
    """Provides option to additionally and automatically
    save figures in a subfolder as a PNG

    Example
    -------


    with MyPdfPages('test.pdf', create_pngs=True) as pdf:
        figh = plt.figure()
        plt.scatter([1, 2, 3, 4, 5], [2, 5, 4, 1, 7])
        pdf.savefig(figh)
    """
    def __init__(self, filename, create_pngs=True, **kwargs):
        self.create_pngs = create_pngs
        if create_pngs:
            folder, fn = os.path.split(filename)
            self.base_name = fn.replace('.pdf', '')
            self.png_folder = opj(folder, self.base_name)
            if not os.path.isdir(self.png_folder):
                os.makedirs(self.png_folder)
            self.page_num = 1
        super().__init__(filename, **kwargs)

    def savefig(self, figure=None, **kwargs):
        if self.create_pngs:
            if not isinstance(figure, Figure):
                if figure is None:
                    manager = Gcf.get_active()
                else:
                    manager = Gcf.get_fig_manager(figure)
                if manager is None:
                    raise ValueError("No figure {}".format(figure))
                figh = manager.canvas.figure
            else:
                figh = figure
            figh.savefig(opj(self.png_folder, '%s_%d.png' % (self.base_name, self.page_num)),
                         format='png',
                         dpi=200)
            self.page_num += 1
        super().savefig(figure=figure, **kwargs)
        
def _test_create_pngs():
    with MyPdfPages('test.pdf', create_pngs=True) as pdf:
        figh = plt.figure()
        plt.scatter([1, 2, 3, 4, 5], [2, 5, 4, 1, 7])
        pdf.savefig(figh)