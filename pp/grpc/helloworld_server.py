import logging
from concurrent import futures

import numpy as np
import torch
import torch.nn as nn 

import base64

import grpc
import helloworld_pb2
import helloworld_pb2_grpc
from helloworld_pb2 import HelloReply, ModelReply
from helloworld_pb2_grpc import GreeterServicer, ModelServicer


class Greeter(GreeterServicer):
    '''
    '''
    def SayHello(self, request, context):
        return HelloReply(message=f'hellow {request.name}')

    def SayHelloAgain(self, request, context):
        return HelloReply(message=f'hello {request.name}')


class Model(ModelServicer):
    '''model
    '''
    def Inference(self, request, context):
        
        a = request.a
        b = request.b
        data = base64.b64decode(request.data)
        
        data = np.frombuffer(data).reshape(int(a), int(b))
        mean = data.mean()
        c = a + b

        return ModelReply(data=base64.b64encode(mean), c=c)



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    helloworld_pb2_grpc.add_ModelServicer_to_server(Model(), server)

    server.add_insecure_port('[::]:50001')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
