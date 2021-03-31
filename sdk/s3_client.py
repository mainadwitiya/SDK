import boto3
import os
import sys
import pickle
import json
from botocore.exceptions import ClientError
import logging
from boto3.s3.transfer import TransferConfig
from threading import Thread, Lock
from io import BytesIO
import requests




class ProgressPercentage(object):
    def __init__(self, fileobj, filename, fsize=None):
        self._fileobj = fileobj
        self._filename = filename
        if not fsize:
            self._size = fileobj.getbuffer().nbytes
        else:
            self._size = fsize

        self._seen_so_far = 0
        self._lock = Lock()

    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s %s %s %s / %s  (%.2f%%)"
                % (
                    "Uploading file ",
                    self._filename,
                    "to S3 =========>",
                    self._seen_so_far,
                    self._size,
                    percentage,
                )
            )
            sys.stdout.flush()


class S3Client:
    """Boto3 client to S3"""

    def __init__(self):
        presigned_cred = requests.post("http://api.alectio.com/experiments/getcredentials").json()
        # boto3 clients to read / write to S3 bucket
        self.client = boto3.client(
            "s3",
            aws_access_key_id=presigned_cred['KEY'],
            aws_secret_access_key=presigned_cred["SECRET"],
        )
        os.environ["AWS_ACCESS_KEY_ID"] = presigned_cred['KEY']
        os.environ["AWS_SECRET_ACCESS_KEY"] = presigned_cred['SECRET']

    def read(self, bucket_name, object_key, file_format):
        """ Read a file from the S3 bucket containing
        this experiment

        object_key: str.
           object key for this file

        file_format: str.
            format of the file {"pickle", "json"}

        """

        s3_object = self.client.get_object(Bucket=bucket_name, Key=object_key)
        body = s3_object["Body"]

        if file_format == "json":
            jstr = body.read().decode("utf-8")
            content = json.loads(jstr)
        elif file_format == "pickle":
            f = body.read()
            content = pickle.loads(f)
        elif file_format == "txt":
            content = body.read().decode(encoding="utf-8", errors="ignore")
        return content

    def write(self, obj, bucket_name, object_key, file_format):
        """Write an object to S3 bucket
        Mostly used for writing ExperimentData.pkl
        InferenceData.pkl files

        obj: dict | list | string

        bucket_name: name of the s3 bucket
        object_key: str.
            object key in the S3 bucket

        file_format: str.
            format of the file to save the object
            {pickle, json}

        """

        # convert obj to byte string
        if file_format == "pickle":
            bytestr = pickle.dumps(obj)
        elif file_format == "json":
            bytestr = json.dumps(obj)
        elif file_format == "txt":
            bytestr = b"{}".format(obj)

        # @TODO add md5 hash
        # @TODO return success or failure message
        # put in S3
        r = self.client.put_object(Bucket=bucket_name, Key=object_key, Body=bytestr,)

        return

    def multi_part_upload_with_s3(self, obj, bucket_name, object_key, file_format):
        # convert obj to byte string
        if file_format == "pickle":
            bytestr = pickle.dumps(obj)
        elif file_format == "json":
            bytestr = bytes(json.dumps(obj).encode("UTF-8"))
            # self.client.put_object(Bucket=bucket_name, Key=object_key, Body=bytestr,)
            # return
        elif file_format == "txt":
            bytestr = b"{}".format(obj)

        fileobj = BytesIO(bytestr)
        # size = sys.getsizeof(fileobj)
        # print("Size of object =" , size)

        config = TransferConfig(
            multipart_threshold=1024 * 25,
            max_concurrency=10,
            multipart_chunksize=1024 * 25,
            use_threads=True,
        )

        self.client.upload_fileobj(
            Fileobj=fileobj,
            Bucket=bucket_name,
            Key=object_key,
            Callback=ProgressPercentage(fileobj, object_key),
            Config=config,
        )

        return

    def multi_part_upload_file(self, obj, bucket_name, object_key):
        config = TransferConfig(
            multipart_threshold=1024 * 25,
            max_concurrency=10,
            multipart_chunksize=1024 * 25,
            use_threads=True,
        )

        self.client.upload_file(
            obj,
            bucket_name,
            object_key,
            Config=config,
            Callback=ProgressPercentage(obj, object_key, os.path.getsize(obj)),
        )

        return

    def upload_file(self, file_name, bucket, object_name=None):
        """Upload a file to an S3 bucket

        :param file_name: File to upload
        :param bucket: Bucket to upload to
        :param object_name: S3 object name. If not specified then file_name is used
        :return: True if file was uploaded, else False
        """

        # If S3 object_name was not specified, use file_name
        if object_name is None:
            object_name = file_name

        # Upload the file
        s3_client = boto3.client('s3')
        try:
            response = s3_client.upload_file(file_name, bucket, object_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True

    def download_checkpoints(self, bucket_name, project_id, experiment_id, cur_loop, log_dir):
        checkpoints_to_download = list(range(cur_loop))

        for checkpoint in checkpoints_to_download:
            print(f"downloading file ckpt_{checkpoint}")
            self.client.download_file(
                f"{bucket_name}",
                f"{project_id}/{experiment_id}/ckpt_{checkpoint}.pth",
                f"{log_dir}/ckpt_{checkpoint}.pth",
            )
