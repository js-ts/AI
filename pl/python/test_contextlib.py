
from contextlib import contextmanager

import sys

# 0
class SysPathG(object):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path

    def __enter__(self, ) -> None:
        sys.path.insert(0, self.path)
        return self.path

    def __exit__(self, type, value, traceback) -> None:
        _p = sys.path.pop(0)
        print('_p', _p)

with SysPathG('/test') as p:
    print('0-p', p)



# 1
@contextmanager
def syspathg(path):
    sys.path.insert(0, path)
    yield path
    _p = sys.path.pop(0)
    print('_p', _p)

with syspathg('./tst') as p:
    print('1-p', p)
