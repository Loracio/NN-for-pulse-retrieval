try:
    __INPUTOUTPUT_IMPORTED__
except NameError:
    __INPUTOUTPUT_IMPORTED__= False

if not __INPUTOUTPUT_IMPORTED__:
    from .read import load_data, load_and_norm_data
    from processData import process_data
    
__INPUTOUTPUT_IMPORTED__ = True