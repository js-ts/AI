from collections import defaultdict
import importlib


ENGINES = defaultdict(dict)


def register(module):
    '''register
    '''
    assert 'module' in module.__dict__, ''
    assert 'support_types' in module.__dict__, ''
    
    support_types = module.support_types
    module = module.module

    if module.__name__ in ENGINES:
        raise AttributeError(f'name: {module.__name__} already exists.')

    ENGINES[module.__name__]['module'] = module # importlib.import_module(module.__name__) 
    ENGINES[module.__name__]['support_types'] = support_types

    return module



import numpy
@register
class NUMPY():
    module = numpy
    support_types = (numpy.ndarray, numpy.float, numpy.float32, numpy.float64, numpy.int, numpy.bool)


try:
    import cupy
    class CUPY():
        module = cupy 
        support_types = (cupy.ndarray, cupy.float, cupy.float32, cupy.float64, cupy.int, cupy.bool)
except:
    print('Cannot import cupy')