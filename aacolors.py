"""These were taken from weblogolib. There these classes have more functionality,
but I just wanted a way to use these assignments as simple lookup"""

AALPHABET='ACDEFGHIKLMNPQRSTVWY'

class aaGrouping(object):
    """Builds lookup dicts of color and label assignments from the set of colorGroups"""
    def __init__(self,groups,title='',description=''):
        self.groups=groups
        self.labelDict={}
        self.colorDict={}
        for g in groups:
            self.colorDict.update({aa:g.color for aa in g.aas})
            self.labelDict.update({aa:g.label for aa in g.aas})

    def label(self,aa):
        return self.labelDict.get(aa,'')

    def color(self,aa):
        return self.colorDict.get(aa,'black')


class ColorGroup(dict):
    """Builds a lookup dict for the amino-acids specified"""
    def __init__(self,aas,color,label=''):
        dict.__init__(self)
        self.color=color
        self.label=label
        self.aas=aas

nucleotide = aaGrouping([
    ColorGroup("G", "orange"),
    ColorGroup("TU", "red"),
    ColorGroup("C",  "blue"),
    ColorGroup("A",  "green")]) 

base_pairing = aaGrouping([
    ColorGroup("TAU",  "darkorange", "Weak (2 Watson-Crick hydrogen bonds)"),
    ColorGroup("GC",    "blue", "Strong (3 Watson-Crick hydrogen bonds)")])

# From Crooks2004c-Proteins-SeqStr.pdf
hydrophobicity = aaGrouping([
    ColorGroup( "RKDENQ",   "blue", "hydrophilic"),
    ColorGroup( "SGHTAP",   "green", "neutral"  ),
    ColorGroup( "YVMCLFIW", "black",  "hydrophobic") ])

# from makelogo
chemistry = aaGrouping([
  ColorGroup( "GSTYC",  "green",   "polar"),
  ColorGroup( "NQ",      "purple", "neutral"), 
  ColorGroup( "KRH",     "blue",   "basic"),
  ColorGroup( "DE",      "red",    "acidic"),
  ColorGroup("PAWFLIMV", "black",  "hydrophobic") ])   

charge = aaGrouping([
    ColorGroup("KRH", "blue", "Positive" ),
    ColorGroup( "DE", "red", "Negative") ])


taylor = aaGrouping([
    ColorGroup( 'A', '#CCFF00' ),
    ColorGroup( 'C', '#FFFF00' ),
    ColorGroup( 'D', '#FF0000'),
    ColorGroup( 'E', '#FF0066' ),
    ColorGroup( 'F', '#00FF66'),
    ColorGroup( 'G', '#FF9900'),
    ColorGroup( 'H', '#0066FF'),
    ColorGroup( 'I', '#66FF00'),
    ColorGroup( 'K', '#6600FF'),
    ColorGroup( 'L', '#33FF00'),
    ColorGroup( 'M', '#00FF00'),
    ColorGroup( 'N', '#CC00FF'),
    ColorGroup( 'P', '#FFCC00'),
    ColorGroup( 'Q', '#FF00CC'),
    ColorGroup( 'R', '#0000FF'),
    ColorGroup( 'S', '#FF3300'),
    ColorGroup( 'T', '#FF6600'),
    ColorGroup( 'V', '#99FF00'),
    ColorGroup( 'W', '#00CCFF'),
    ColorGroup( 'Y', '#00FFCC')],
    title = "Taylor",
    description = "W. Taylor, Protein Engineering, Vol 10 , 743-746 (1997)")