from __future__ import print_function

import grpc
import alectio_sdk.proto.bidirectional_pb2_grpc as bidirectional_pb2_grpc
import alectio_sdk.proto.bidirectional_pb2 as bidirectional_pb2


GRPCBACKEND = "grpc.alectio.com:50051"
credentials = grpc.ssl_channel_credentials()

class Backend(object):
    def __init__(self, token):
        self.token = token
        self._done = False
        self.result_q = None
        self.response = None

    def make_start_exp_payload(self):
        yield bidirectional_pb2.StartExperiment(
            exp_token=self.token
        )
    
    def make_sdk_response_payload(self):
        yield bidirectional_pb2.GetExperimentResponse(
            exp_token=self.token
        )

    def startExperiment(self):
        # project_id = request.get_json()['project_id']
        # user_id = request.get_json()['user_id']
        # experiment_id = request.get_json()['experiment_id']
        credentials = grpc.ssl_channel_credentials()
        with grpc.secure_channel(GRPCBACKEND, credentials) as channel:
            stub = bidirectional_pb2_grpc.BidirectionalStub(channel)
            responses = stub.GetStartExperimentResponse(self.make_start_exp_payload())
            for response in responses:
                return response.exp_token
    
    def getSDKResponse(self):
        credentials = grpc.ssl_channel_credentials()
        with grpc.secure_channel(GRPCBACKEND, credentials) as channel:
            stub = bidirectional_pb2_grpc.BidirectionalStub(channel)
            responses = stub.GetSDKResponse(self.make_sdk_response_payload())
            response_obj = next(responses)
            response = {
                "status": response_obj.status,
                "experiment_id": response_obj.experiment_id,
                "project_id": response_obj.project_id,
                "cur_loop": response_obj.cur_loop,
                "user_id": response_obj.user_id,
                "bucket_name": response_obj.bucket_name,
                "type": response_obj.type,
                "n_rec": response_obj.n_rec,
                "n_loop": response_obj.n_loop
            }
            return response

def make_message(message):
    return bidirectional_pb2.StartExperiment(
        exp_token=message
    )


def generate_messages():
    messages = [
        make_message("First message"),
        make_message("Second message"),
        make_message("Third message"),
        make_message("Fourth message"),
        make_message("Fifth message"),
    ]
    for msg in messages:
        print("Hello Server Sending you the %s" % msg.exp_token)
        yield msg


def send_message(stub):
    responses = stub.GetStartExperimentResponse(generate_messages())
    for response in responses:
        print("Hello from the server received your %s" % response.exp_token)


def run():
    credentials = grpc.ssl_channel_credentials()
    with grpc.secure_channel(GRPCBACKEND, credentials) as channel:
        stub = bidirectional_pb2_grpc.BidirectionalStub(channel)
        send_message(stub)


if __name__ == '__main__':
    #run()
    backend = Backend("03af80240af4084af08320adf03040")
    backend.startExperiment()
