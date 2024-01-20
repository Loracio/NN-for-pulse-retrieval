try:
    __VISUALIZATION_IMPORTED__
except NameError:
    __VISUALIZATION_IMPORTED__= False

if not __VISUALIZATION_IMPORTED__:
    from .visualization import results_GUI
    
__VISUALIZATION_IMPORTED__ = True