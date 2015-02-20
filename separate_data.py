import pickle

__all__ = ['SeparateData']

class SeparateData(object):
    """Object methods for (un)pickling only the data attributes of a class as a dict

    Methods
    -------
    to_dict : Return a dict representation of the data members of the class
    from_dict : Copy values from a dict into an instance of the class
    saveFile : Pickle and save the dict to a file
    loadFile : Instantiate a new class and load in data from the file"""
    def to_dict(self,skip=[],special={}):
        out = {}
        for attr in dir(self):
            if attr in skip:
                pass
            elif hasattr(self.__getattribute__(attr),'__call__'):
                pass
            elif attr[:2] == '__':
                pass
            elif attr in special.keys():
                out[attr] = special[attr](self.__getattribute__(attr))
            else:
                out[attr] = self.__getattribute__(attr)
        return out
    def from_dict(self,d,special={},kwargs={}):
        for k in kwargs.keys():
            self.__setattr__(k,kwargs[k])
        for k in d.keys():
            if not k in special.keys():
                self.__setattr__(k,d[k])
            else:
                self.__setattr__(k,special[k](d[k]))
    def saveFile(self,filename):
        with open(filename,'wb') as filehandle:
            pickle.dump(self.to_dict(),filehandle)
    def loadFile(self,filename,**kwargs):
        with open(filename,'rb') as filehandle:
            obj = pickle.load(filehandle)
        self.from_dict(obj,kwargs=kwargs)
    def fromOldInstance(self,old):
        self.from_dict(old.to_dict(skip=[]))

def test_SD():
    """Used for testing the SeparateData class.
    TODO: Make it a real test function rather than printing indicators"""
    import pandas as pd
    import tempfile
    class SDSub(SeparateData):
        def __init__(self):
            self.keep1 = rand(3)
            self.keep2 = [1,2,3]
            self.keep3 = pd.DataFrame(rand(2,3))

            self.skip1 = rand(2)
            self.skip2 = [8,9,10]

    s = SDSub()
    out = s.to_dict()
    out_skipped = s.to_dict(skip=['skip1','skip2'])

    s2 = SDSub()
    #print s2.keep1,s2.skip1
    s2.from_dict(out_skipped)
    #print s2.keep1,s2.skip1
    s2.from_dict(out,kwargs=dict(skip1=rand(4),skip2=[10,11]))
    #print s2.keep1,s2.skip1

    fn = tempfile.mktemp()
    s.saveFile(fn)
    fn_skip = tempfile.mktemp()
    s.saveFile(fn_skip,skip=['skip1','skip2'])

    s3 = SDSub()
    #print s3.keep1,s3.skip1
    s3.loadFile(fn_skip)
    #print s3.keep1,s3.skip1
    s3.loadFile(fn,skip1=rand(4),skip2=[10,11])
    #print s3.keep1,s3.skip1