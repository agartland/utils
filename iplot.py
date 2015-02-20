"""
Interactive plotting tools for matplotlib

Adapted from the scipy.org cookbooks on interactive matplotlib plots.
http://wiki.scipy.org/Cookbook/Matplotlib/Interactive_Plotting
"""

import math
import pylab

__all__ = ['AnnotationPicker',
           'linkPickers',
           'AnnoteFinder']

class AnnotationPicker:
    """
    Callback for matplotlib to display an annotation when points are clicked on.
        
    Register this function like this:
        
    scatter(xdata, ydata,picker=5) #for 5 points of tolerance
    mp = AnnotationPicker(xdata, ydata, notes)
    """

    def __init__(self, xdata, ydata, notes, axis=None,**kwargs):
        self.xdata = xdata
        self.notes=notes
        self.ydata=ydata

        if axis is None:
            self.axis = pylab.gca()
        else:
            self.axis= axis
        self.fig=self.axis.figure
        
        self.drawn = {}
        self.connect()
        self.links=[]
        self.textKwargs=kwargs

    def callback(self, event):
        for ind in event.ind:
            self.drawOne(ind)
            for linkedPicker in self.links:
                linkedPicker.drawOne(ind)
    def drawOne(self,ind):
        x,y=self.xdata[ind],self.ydata[ind]
        note=self.notes[ind]
        if (x,y) in self.drawn.keys():
            markers = self.drawn[(x,y)]
            for m in markers:
                m.set_visible(not m.get_visible())
            self.axis.figure.canvas.draw()
        else:
            kwargs={'xy':(x,y),
                    'size':'x-small',
                    'family':'monospace',
                    'xytext':(5,5),
                    'textcoords':'offset points'}
            kwargs.update(self.textKwargs)
            t = self.axis.annotate( "%s" % (note),**kwargs)
            m = self.axis.scatter([x],[y], marker='d', c='r', zorder=100,s=1)
            self.drawn.update({(x,y):(t,m)})
            self.fig.canvas.draw()
        
    def __del__(self):
        self.disconnect()
    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.connid)
    def connect(self):
        self.connid=self.fig.canvas.mpl_connect('pick_event',self.callback)

def linkPickers(pickers):
    """Links a sequence of pickers such that picking a point on one plot
    will activate the picker in the other plot as well"""
    for i in range(len(pickers)):
        allButSelfPickers = pickers[:i]+pickers[i+1:]
        pickers[i].links.extend(allButSelfPickers)

class AnnoteFinder:
    """
    Callback for matplotlib to display an annotation when points are clicked on.
    The point which is closest to the click and within xtol and ytol is identified.
        
    Register this function like this:
        
    scatter(xdata, ydata)
    af = AnnoteFinder(xdata, ydata, annotes)
    connect('button_press_event', af)
    """

    def __init__(self, xdata, ydata, annotes, axis=None, xtol=None, ytol=None,coordCaption=False):
        self.data = zip(xdata, ydata, annotes)
        if xtol is None:
            xtol = ((max(xdata) - min(xdata))/float(len(xdata)))/2
        if ytol is None:
            ytol = ((max(ydata) - min(ydata))/float(len(ydata)))/2
        self.xtol = xtol
        self.ytol = ytol
        if axis is None:
            self.axis = pylab.gca()
        else:
            self.axis= axis
        self.drawnAnnotations = {}
        self.links = []
        self.coordCaption=coordCaption

    def distance(self, x1, x2, y1, y2):
        """Return the distance between two points"""
        return math.hypot(x1 - x2, y1 - y2)

    def __call__(self, event):
        if event.inaxes:
            clickX = event.xdata
            clickY = event.ydata
            if self.axis is None or self.axis==event.inaxes:
                annotes = []
                for x,y,a in self.data:
                    if clickX-self.xtol < x < clickX+self.xtol and clickY-self.ytol < y < clickY+self.ytol :
                        annotes.append((self.distance(x,clickX,y,clickY),x,y, a) )
                if annotes:
                    annotes.sort()
                    distance, x, y, annote = annotes[0]
                    self.drawAnnote(event.inaxes, x, y, annote)
                    for l in self.links:
                        l.drawSpecificAnnote(annote)

    def drawAnnote(self, axis, x, y, annote):
        """
        Draw the annotation on the plot
        """
        if (x,y) in self.drawnAnnotations:
            markers = self.drawnAnnotations[(x,y)]
            for m in markers:
                m.set_visible(not m.get_visible())
            self.axis.figure.canvas.draw()
        else:
            kwargs={'xy':(x,y),
                    'size':'small',
                    'family':'monospace',
                    'xytext':(5,5),
                    'textcoords':'offset points'}
            if self.coordCaption:
                t = axis.annotate("(%3.2f, %3.2f) - %s"%(x,y,annote), **kwargs)
            else:
                t = axis.annotate( "%s" % (annote),**kwargs)
            m = axis.scatter([x],[y], marker='d', c='r', zorder=100,s=1)
            self.drawnAnnotations[(x,y)] =(t,m)
            self.axis.figure.canvas.draw()

    def drawSpecificAnnote(self, annote):
        annotesToDraw = [(x,y,a) for x,y,a in self.data if a==annote]
        for x,y,a in annotesToDraw:
            self.drawAnnote(self.axis, x, y, a)