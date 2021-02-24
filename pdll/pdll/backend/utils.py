
from .engine import ENGINES

from . import np, support_types 


def set_engine(name='numpy'):
    '''
    '''
    global np, support_types

    assert name in ENGINES, f'{name} not registe.'

    np = ENGINES[name]['module']
    support_types = ENGINES[name]['support_types']

    # return np, support_types
