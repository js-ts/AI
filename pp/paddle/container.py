
import paddle

import operator
import itertools

from typing import Union, Iterable, TypeVar



class Sequential(paddle.nn.Sequential):
    '''
    '''
    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(itertools.islice(iterator, idx, None))
        
    def __getitem__(self, idx: Union[slice, int, str]):
        r'''get 
        mm is sequential instance
        mm[1]
        mm[-1]
        mm[1:]
        mm['L1']
        '''
        if isinstance(idx, str):
            return self._sub_layers[idx]
        elif isinstance(idx, slice):
            return self.__class__(*list(self._sub_layers.items())[idx])
        else:
            return self._get_item_by_idx(self._sub_layers.values(), idx)
        
    def __setitem__(self, idx: Union[int, str], layer: paddle.nn.Layer) -> None:
        r'''set
        mm is sequential instance
        mm[1] = `Layer Instance`
        mm['L1'] = `Layer Instance`
        '''
        if isinstance(idx, str):
            return setattr(self, str(idx), layer)
        else:
            key = self._get_item_by_idx(self._sub_layers.keys(), idx)
            return setattr(self, key, layer)
        
    def __delitem__(self, idx: Union[slice, int, str]) -> None:
        r'''del 
        mm is sequential instance
        del mm[1]
        del mm[-1]
        del mm[1:]
        del mm['L1']
        '''
        if isinstance(idx, slice):
            for key in list(self._sub_layers.keys())[idx]:
                delattr(self, key)
        elif isinstance(idx, int):
            key = self._get_item_by_idx(self._sub_layers.keys(), idx)
            delattr(self, key)
        else:
            delattr(self, idx)



class LayerList(paddle.nn.LayerList):
    '''
    '''
    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx: Union[slice, int]):
        if isinstance(idx, slice):
            return self.__class__(list(self._sub_layers.values())[idx])
        else:
            return self._sub_layers[self._get_abs_string_index(idx)]
        
    def __setitem__(self, idx: int, layer: paddle.nn.Layer) -> None:
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), layer)
    
    # T = TypeVar('T')
    def __iadd__(self, layers: Iterable[paddle.nn.Layer]):
        return self.extend(layers)
