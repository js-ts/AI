import logging

import base64
import numpy as np 

import grpc
import helloworld_pb2
import helloworld_pb2_grpc
from helloworld_pb2 import HelloRequest, ModelRequest
from helloworld_pb2_grpc import GreeterStub, ModelStub

def run():
    '''
    '''
    with grpc.insecure_channel('localhost:50001') as channel:
        # stub = GreeterStub(channel)
        # response = stub.SayHello(HelloRequest(name='you'))
        # print(response.message)

        stub = ModelStub(channel)

        # modelRequest = ModelRequest()
        # modelRequest.a = 10
        # modelRequest.b = 20
        # modelRequest.data = base64.b64encode(np.random.rand(a, b))

        a = 10
        b = 20
        data = base64.b64encode(np.random.rand(a, b))

        response = stub.Inference(ModelRequest(data=data, a=a, b=b))

        print(response.data, response.c)

        mean = np.frombuffer(base64.b64decode(response.data))

        print(type(response), type(response.c))
        print(mean)


if __name__ == '__main__':
    logging.basicConfig()

    run()
    