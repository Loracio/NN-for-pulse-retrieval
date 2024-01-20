try:
    __UTILS_IMPORTED__
except NameError:
    __UTILS_IMPORTED__= False

if not __UTILS_IMPORTED__:
    from .utils import compute_trace, compute_trace_error, DFT, IDFT
    
__UTILS_IMPORTED__ = True