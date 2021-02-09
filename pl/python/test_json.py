import json
import numpy as np
import io
import base64

arr = np.random.rand(3, 3)
var = np.random.rand(1)

data = {
    'a': 10,
    'arr': arr.tolist(),
    'var': var.item()
}

print(json.loads(json.dumps(data)))



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj): # pylint: disable=E0202
        '''
        '''
        if isinstance(obj, np.ndarray): 
            out = io.BytesIO()
            np.savez_compressed(out, obj=obj)
            # b64str = base64.b64encode(out.getvalue()).decode()
            b64str = str(base64.b64encode(out.getvalue()),  encoding='utf-8')
            return {'b64str': b64str}

        return json.JSONEncoder.default(self, obj)


def NumpyDecoderHook(dct):
    '''
    '''
    if isinstance(dct, dict) and 'b64str' in dct:
        # b64bytes = dct['b64str'].encode()
        b64bytes = bytes(dct['b64str'], encoding='utf-8')
        output = io.BytesIO(base64.b64decode(b64bytes))
        output.seek(0)
        return np.load(output)['obj']

    return dct


data = {
    'a': 10,
    'arr': arr,
    'var': var
}

dumped = json.dumps(data, cls=NumpyEncoder)

out = json.loads(dumped, object_hook=NumpyDecoderHook)

print('\n', out, '\n')
print(type(out['arr']), out['arr'].dtype, out['arr'].shape)