
from concurrent.futures import ThreadPoolExecutor
import grpc


from cb_pb2 import PredictRequest, PredictResponse
from cb_pb2_grpc import Inference, InferenceStub, InferenceServicer, add_InferenceServicer_to_server

from predict import predict


class CatboostInferenceService(InferenceServicer):

    def Predict(self, request: PredictRequest, context):
        # print("Request", request)
        reply = predict(request)
        # print("Reply", reply)
        return reply


def serve():
    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    add_InferenceServicer_to_server(CatboostInferenceService(), server)
    server.add_insecure_port("0.0.0.0:50052")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
