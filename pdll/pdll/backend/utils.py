


def set_device(name='cpu', engine='numpy'):
    '''
    '''
    if name.lower() == 'cpu' and engine == 'numpy':
        from .numpy import np, support_types
        return np, support_types
