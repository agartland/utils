import numpy as np

__all__ = ['HVTNcolors', 'HVTNrx', 'HVTNLU']

HVTNcolors = np.array([(120, 120, 115),
                       (23, 73, 255),
                       (217, 35, 33),
                       (10, 183, 201),
                       (255, 111, 27),
                       (129, 0, 148),
                       (255, 94, 191),
                       (143, 143, 143)])/255.
HVTNrx = ['C', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Negative']

HVTNLU = {rx:c for rx,c in zip(HVTNrx, HVTNcolors)}