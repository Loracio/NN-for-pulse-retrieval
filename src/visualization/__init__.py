try:
    __VISUALIZATION_IMPORTED__
except NameError:
    __VISUALIZATION_IMPORTED__= False

if not __VISUALIZATION_IMPORTED__:
    from .visualization import resultsGUI
    
__VISUALIZATION_IMPORTED__ = True