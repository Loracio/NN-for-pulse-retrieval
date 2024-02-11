try:
    __UTILS_IMPORTED__
except NameError:
    __UTILS_IMPORTED__= False

if not __UTILS_IMPORTED__:
    from .utils import fourier_utils, meanVal
    
__UTILS_IMPORTED__ = True