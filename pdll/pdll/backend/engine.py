from collections import OrderedDict
import importlib


ENGINES = OrderedDict()

def register(module):
    '''register
    '''
    if module.__name__ in ENGINES:
        raise AttributeError(f'name: {module.__name__} already exists.')

    # importlib.import_module(module.__name__)
    assert 'support_types' in module.__dict__, 'module must have `support_types` attr.'
    
    ENGINES[module.__name__]['module'] = module
    ENGINES[module.__name__]['support_types'] = module.support_types

    return module