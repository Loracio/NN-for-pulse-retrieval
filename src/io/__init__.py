try:
    __INPUTOUTPUT_IMPORTED__
except NameError:
    __INPUTOUTPUT_IMPORTED__= False

if not __INPUTOUTPUT_IMPORTED__:
    from .read import load_data, load_and_norm_data, read_tfrecord
    from .processData import process_data, process_data_tfrecord
    
__INPUTOUTPUT_IMPORTED__ = True