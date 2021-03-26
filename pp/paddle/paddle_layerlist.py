

import paddle
from collections import OrderedDict
import paddle.fluid as fluid
import unittest
import numpy as np 



class LayerList(paddle.nn.LayerList):

    def _get_abs_idx(self, idx):
        if isinstance(idx, int):
            if not (-len(self) <= idx < len(self)):
                raise IndexError('index {} is out of range, should be an integer in range [{}, {})'.format(idx, -len(self), len(self)))
            if idx < 0:
                idx += len(self)
        return idx

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._sub_layers.values())[idx])
        else:
            idx = self._get_abs_idx(idx)
            return self._sub_layers[str(idx)]

    def __setitem__(self, idx, sublayer):
        idx = self._get_abs_idx(idx)
        return setattr(self, str(idx), sublayer)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._sub_layers))[idx]:
                delattr(self, str(k))
        else:
            idx = self._get_abs_idx(idx)
            delattr(self, str(idx))
        str_indices = [str(i) for i in range(len(self._sub_layers))]
        self._sub_layers = OrderedDict(
            list(zip(str_indices, self._sub_layers.values())))

    def __len__(self):
        return len(self._sub_layers)

    def __iter__(self):
        return iter(self._sub_layers.values())

    def append(self, sublayer):
        """
        Appends a sublayer to the end of the list.

        Parameters:
            sublayer (Layer): sublayer to append

        Examples:
            .. code-block:: python

                import paddle

                linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
                another = paddle.nn.Linear(10, 10)
                linears.append(another)
                print(len(linears))  # 11
        """
        self.add_sublayer(str(len(self)), sublayer)
        return self

    def insert(self, index, sublayer):
        """
        Insert a sublayer before a given index in the list.

        Parameters:
            index (int): index to insert.
            sublayer (Layer): sublayer to insert

        Examples:
            .. code-block:: python

                import paddle

                linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
                another = paddle.nn.Linear(10, 10)
                linears.insert(3, another)
                print(linears[3] is another)  # True
        """
        assert isinstance(index, int) and \
               -len(self._sub_layers) <= index < len(self._sub_layers), \
            "index should be an integer in range [{}, {})".format(-len(self), len(self))

        index = self._get_abs_idx(index)
        for i in range(len(self._sub_layers), index, -1):
            self._sub_layers[str(i)] = self._sub_layers[str(i - 1)]
        self._sub_layers[str(index)] = sublayer

    def extend(self, sublayers):
        """
        Appends sublayers to the end of the list.

        Parameters:
            sublayers (iterable of Layer): iterable of sublayers to append

        Examples:
            .. code-block:: python

                import paddle

                linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
                another_list = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(5)])
                linears.extend(another_list)
                print(len(linears))  # 15
                print(another_list[0] is linears[10])  # True
        """
        offset = len(self)
        for i, sublayer in enumerate(sublayers):
            idx = str(offset + i)
            self.add_sublayer(idx, sublayer)
        return self




class MyLayer(fluid.Layer):
    def __init__(self, layerlist):
        super(MyLayer, self).__init__()
        self.layerlist = layerlist

    def forward(self, x):
        for l in self.layerlist:
            x = l(x)
        return x


class TestImperativeContainer(unittest.TestCase):
    def fluid_dygraph_list(self):
        return LayerList(
            [fluid.dygraph.Linear(2**i, 2**(i + 1)) for i in range(6)])

    def paddle_imperative_list(self):
        return LayerList(
            [fluid.dygraph.Linear(2**i, 2**(i + 1)) for i in range(6)])

    def layer_list(self, use_fluid_api):
        data_np = np.random.uniform(-1, 1, [5, 1]).astype('float32')
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(data_np)
            layerlist = self.fluid_dygraph_list(
            ) if use_fluid_api else self.paddle_imperative_list()
            size = len(layerlist)

            model = MyLayer(layerlist)
            res1 = model(x)
            self.assertListEqual(res1.shape, [5, 2**size])
            model.layerlist[size - 1] = fluid.dygraph.Linear(2**(size - 1), 5)
            res2 = model(x)
            self.assertListEqual(res2.shape, [5, 5])
            del model.layerlist[size - 1]
            res3 = model(x)
            self.assertListEqual(res3.shape, [5, 2**(size - 1)])
            model.layerlist.append(fluid.dygraph.Linear(2**(size - 1), 3))
            res4 = model(x)
            self.assertListEqual(res4.shape, [5, 3])
            res4.backward()

            model2 = MyLayer(layerlist[:-1])
            res5 = model2(x)
            self.assertListEqual(res5.shape, [5, 2**(size - 1)])
            del model2.layerlist[1:]
            res6 = model2(x)
            self.assertListEqual(res6.shape, [5, 2**(0 + 1)])
            res6.backward()

            model3 = MyLayer(layerlist[:-2])
            model3.layerlist.append(fluid.dygraph.Linear(3, 1))
            model3.layerlist.insert(size - 2,
                                    fluid.dygraph.Linear(2**(size - 2), 3))
            res7 = model3(x)
            self.assertListEqual(res7.shape, [5, 1])
            to_be_extended = [
                fluid.dygraph.Linear(3**i, 3**(i + 1)) for i in range(3)
            ]
            model3.layerlist.extend(to_be_extended)
            res8 = model3(x)
            self.assertListEqual(res8.shape, [5, 3**3])
            res8.backward()

            model4 = MyLayer(layerlist[:3])
            model4.layerlist[-1] = fluid.dygraph.Linear(4, 5)
            res9 = model4(x)
            self.assertListEqual(res9.shape, [5, 5])
            del model4.layerlist[-1]
            res10 = model4(x)
            self.assertListEqual(res10.shape, [5, 4])
            model4.layerlist.insert(-1, fluid.dygraph.Linear(2, 2))
            res11 = model4(x)
            self.assertListEqual(res11.shape, [5, 4])
            res11.backward()
            
            print(model4.layerlist['1'])

    def test_layer_list(self):
        self.layer_list(True)
        self.layer_list(False)


if __name__ == '__main__':
    unittest.main()
