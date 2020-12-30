import pandas as pd
# from waitress import serve
import numpy as np
import requests
import traceback
import sys
import os
import psutil
import time
import boto3
import logging
import sklearn.metrics
from copy import deepcopy
from .s3_client import S3Client
from alectio_sdk.metrics.object_detection import Metrics, batch_to_numpy
from alectio_sdk.backend.backend import Backend
import sentry_sdk
# modules for testing
import argparse
import json


class Pipeline(object):
    r"""
    A wrapper for your `train`, `test`, and `infer` function. The arguments for your functions should be specifed
    separately and passed to your pipeline object during creation.

    Args:
        name (str): experiment name
        train_fn (function): function to be executed in the train cycle of the experiment.
        test_fn (function): function to be executed in the test cycle of the experiment.
        infer_fn (function): function to be executed in the inference cycle of the experiment.
        getstate_fn (function): function specifying a mapping between indices and file names.

    """

    def __init__(self, name, train_fn, test_fn, infer_fn, getstate_fn, args, token):
        """
        """
        self.logger = logging.getLogger("alectio logger")
        self.ch = logging.StreamHandler()

        self.ch.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.ch.setFormatter(self.formatter)
        self.logger.addHandler(self.ch)

        self.train_fn = train_fn
        self.test_fn = test_fn
        self.infer_fn = infer_fn
        self.getstate_fn = getstate_fn
        self.args = args
        self.client = S3Client()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "config.json"), "r") as f:
            self.config = json.load(f)

        self.client_token = token
        self.backend = None

    def _notifyserverstatus(self, logdir):
        logging.basicConfig(
            filename=os.path.join(logdir, "Appstatus.log"), level=logging.INFO
        )
        self.logger.info("Alectio experiment initialized successfully")
        self.logger.info(
            "Training checkpoints and other logs for current experiment will be written into the folder {}".format(
                logdir
            )
        )
        self.logger.info(
            "Press CTRL + C to exit the experiment , if this run terminates in the middle just relaunch alectio sdk"
        )

    def _checkdirs(self, dir_):
        if not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)


    def _estimate_exp_time(self, last_time):
        """
        Estimates the compute time remaining for the experiment

        Args:
            train_times (list): training_times noted down so far
            n_loop (int): total number of loops
        """
        def convert(seconds):
            seconds = seconds % (24 * 3600)
            hour = seconds // 3600
            seconds %= 3600
            minutes = seconds // 60
            seconds %= 60

            return "%d:%02d:%02d" % (hour, minutes, seconds)

        loops_completed = self.cur_loop + 1
        time_left = convert(last_time * (self.n_loop - loops_completed))
        self.logger.info("Estimated time left for the experiment: {}".format(time_left))
        return time_left

    def one_loop(self, request):
        # Get payload args

        self.logger.info("Extracting payload arguments from Alectio")
        # Get payload args
        payload = {
            "experiment_id": request["experiment_id"],
            "project_id": request["project_id"],
            "cur_loop": request["cur_loop"],
            "user_id": request["user_id"],
            "bucket_name": request["bucket_name"],
            "type": request["type"],
            "n_rec": request["n_rec"],
            "n_loop": request["n_loop"]
            }
        print('Current payload: ')
        print(payload)

        self.logdir = payload["experiment_id"]
        self._checkdirs(self.logdir)
        self.args["LOG_DIR"] = self.logdir

        self._notifyserverstatus(self.logdir)
        self.logger.info("Valid payload arguments extracted")
        self.logger.info("Initializing process to train and optimize your model")
        returned_payload = self._one_loop(payload, self.args)
        self.logger.info("Optimization process complete !")
        self.logger.info(
            "Your results for this loop should be visible in Alectio website shortly"
        )

        backend_ip = self.config["backend_ip"]
        port = 80
        url = "".join(["http://", backend_ip, ":{}".format(port), "/end_of_task"])

        headers = {"Authorization": "Bearer " + self.client_token}
        status = requests.post(
            url=url, json=returned_payload, headers=headers
        ).status_code
        if status == 200:
            self.logger.info(
                "Experiment {} running".format(payload["experiment_id"])
            )
            return {"Message": "Loop Started - 200 status returned"}
        else:
            return {"Message": "Loop Failed - non 200 status returned"}

    def _one_loop(self, payload, args):
        r"""
        Executes one loop of active learning. Returns the read `payload` back to the user.

        Args:
           args: a dict with the key `sample_payload` (required path) and any arguments needed by the `train`, `test`
           and infer functions.
        Example::

            args = {sample_payload: 'sample_payload.json', EXPT_DIR : "./log", exp_name: "test", \
                                                                 train_epochs: 1, batch_size: 8}
            app._one_loop(args)

        """

        # payload = json.load(open(args["sample_payload"]))
        self.logger.info("Extracting essential experiment params")

        # read selected indices upto this loop
        payload["cur_loop"] = int(payload["cur_loop"])
        # self.curout_loop = payload["cur_loop"]

        self.cur_loop = payload["cur_loop"]
        self.bucket_name = payload["bucket_name"]
        self.n_loop = payload["n_loop"]

        # type of the ML problem
        self.type = payload["type"]

        # dir for expt log in S3
        #expt_dir = [payload["user_id"], payload["project_id"], payload["experiment_id"]] # This was never being used

        if self.bucket_name == self.config["sandbox_bucket"]:
            # shared S3 bucket for sandbox user
            self.expt_dir = os.path.join(
                payload["user_id"], payload["project_id"], payload["experiment_id"]
            )

            self.project_dir = os.path.join(payload["user_id"], payload["project_id"])

        else:
            # dedicated S3 bucket for paid user
            self.expt_dir = os.path.join(
                payload["project_id"], payload["experiment_id"]
            )

            self.project_dir = os.path.join(payload["project_id"])

        self.logger.info("Essential experiment params have been extracted")
        # get meta-data of the data set
        self.logger.info("Verifying the meta.json that was uploaded by the user")
        key = os.path.join(self.project_dir, "meta.json")
        bucket = boto3.resource("s3").Bucket(self.bucket_name)

        json_load_s3 = lambda f: json.load(bucket.Object(key=f).get()["Body"])
        self.meta_data = json_load_s3(key)
        self.logger.info(
            "SDK Retrieved file: {} from bucket : {}".format(key, self.bucket_name)
        )

        # self.meta_data = self.client.read(self.bucket_name, key, "json")
        # logging.info('SDK Retrieved file: {} from bucket : {}'.format(key, self.bucket_name))

        if self.cur_loop == 0:
            self.resume_from = None
            self.logger.info(
                "Extracting indices for our reference, this may take time ... Please be patient"
            )
            self.state_json = self.getstate_fn(args)
            object_key = os.path.join(self.expt_dir, "data_map.pkl")
            self.logger.info("Extraction complete !!!")
            self.logger.info(
                "Creating index to records data reference for the current experiment"
            )
            self.client.multi_part_upload_with_s3(
                self.state_json, self.bucket_name, object_key, "pickle"
            )
            self.logger.info("Reference creation complete")
        else:

            # check if ckpt cur_loop - 1 exists, otherwise we need to download it from S3
            if not os.path.isfile(
                os.path.join(self.args["EXPT_DIR"], f"ckpt_{self.cur_loop-1}")
            ):
                # need to download the checkpoint files from S3
                self.logger.info(
                    "Starting to copy checkpoints for cloned experiment..."
                )
                self.client.download_checkpoints(
                    payload["bucket_name"],
                    payload["project_id"],
                    payload["experiment_id"],
                    payload["cur_loop"],
                    self.args["EXPT_DIR"],
                )
                self.logger.info(
                    "Finished downloading checkpoints for cloned experiment"
                )

            self.logger.info("Resuming from a checkpoint from a previous loop ")
            # two dag approach needs to refer to the previous checkpoint
            self.resume_from = "ckpt_{}".format(self.cur_loop - 1)

        self.ckpt_file = "ckpt_{}".format(self.cur_loop)
        self.logger.info("Initializing training of your model")

        self.train(args)
        self.logger.info("Training complete !")
        self.logger.info("Initializing testing of your model !")
        self.test(args)
        self.logger.info("Testing complete !")
        self.logger.info("Assessing current best model")
        self.infer(args)
        self.logger.info("Assesment complete ")
        self.logger.info(
            "Time to check what records to train on next loop , visit our front end for more details"
        )

        # Drop unwanted payload values
        del payload["type"]
        del payload["cur_loop"]
        del payload["bucket_name"]
        payload['sdk_version'] = '0.6.8'
        return payload

    def train(self, args):
        r"""
        A wrapper for your `train` function. Returns `None`.

        Args:
           args: a dict whose keys include all of the arguments needed for your `train` function which is defined in `processes.py`.

        """
        start = time.time()

        self.labeled = []
        self.logger.info("Reading indices to train on")
        for i in range(self.cur_loop + 1):
            object_key = os.path.join(
                self.expt_dir, "selected_indices_{}.pkl".format(i)
            )
            selected_indices = self.client.read(
                self.bucket_name, object_key=object_key, file_format="pickle"
            )
            self.labeled.extend(selected_indices)

        self.logger.info("Labelled records are ready to be trained")
        self.labeled.sort()  # Maintain increasing order

        train_op = self.train_fn(
                                args,
                                labeled=deepcopy(self.labeled),
                                resume_from=self.resume_from,
                                ckpt_file=self.ckpt_file,
                                )

        if train_op is not None:
            labels = train_op["labels"]
            unique, counts = np.unique(labels, return_counts=True)
            num_queried_per_class = {u:c for u, c in zip(unique, counts)}

        end = time.time()

        # @TODO compute insights from labels
        if train_op is not None:
            insights = {"train_time": end - start,"num_queried_per_class": num_queried_per_class}
        else:
            insights = {"train_time": end - start}

        self._estimate_exp_time(insights["train_time"])
        object_key = os.path.join(
            self.expt_dir, "insights_{}.pkl".format(self.cur_loop)
        )

        self.client.multi_part_upload_with_s3(
            insights, self.bucket_name, object_key, "pickle"
        )

        return

    def test(self, args):
        r"""
        A wrapper for your `test` function which writes predictions and ground truth to the specified S3 bucket. Returns `None`.

        Args:
           args: a dict whose keys include all of the arguments needed for your `test` function which is defined in `processes.py`.

        """
        self.logger.info("Extracting test results ")
        res = self.test_fn(args, ckpt_file=self.ckpt_file)

        predictions, ground_truth = res["predictions"], res["labels"]
        self.logger.info("Writing test results to S3")

        # write predictions and labels to S3
        object_key = os.path.join(
            self.expt_dir, "test_predictions_{}.pkl".format(self.cur_loop)
        )
        self.client.multi_part_upload_with_s3(
            predictions, self.bucket_name, object_key, "pickle"
        )

        if self.cur_loop == 0:
            # write ground truth to S3
            object_key = os.path.join(
                self.expt_dir, "test_ground_truth_{}.pkl".format(self.cur_loop)
            )
            self.client.multi_part_upload_with_s3(
                ground_truth, self.bucket_name, object_key, "pickle"
            )

        self.compute_metrics(predictions, ground_truth)
        return

    def compute_metrics(self, predictions, ground_truth):
        metrics = {}
        if self.type == "Object Detection":
            det_boxes, det_labels, det_scores, true_boxes, true_labels = batch_to_numpy(
                predictions, ground_truth
            )

            m = Metrics(
                det_boxes=det_boxes,
                det_labels=det_labels,
                det_scores=det_scores,
                true_boxes=true_boxes,
                true_labels=true_labels,
                num_classes=len(self.meta_data["class_labels"]),
            )

            metrics = {
                "mAP": m.getmAP(),
                "AP": m.getAP(),
                "precision": m.getprecision(),
                "recall": m.getrecall(),
                "confusion_matrix": m.getCM().tolist(),
                "class_labels": self.meta_data["class_labels"],
            }

        if self.type == "Classification" or self.type == "Text Classification" or self.type == "Image Classification":
            confusion_matrix = sklearn.metrics.confusion_matrix(
                ground_truth, predictions
            )
            acc_per_class = {
                k: v.round(3)
                for k, v in enumerate(
                    confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
                )
            }
            accuracy = sklearn.metrics.accuracy_score(ground_truth, predictions)
            FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
            FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            TP = confusion_matrix.diagonal()
            TN = confusion_matrix.sum() - (FP + FN + TP)
            precision = TP / (TP + FP)
            precision[np.isnan(precision)] = 0
            recall = TP / (TP + FN)
            recall[np.isnan(recall)] = 0
            f1_score = 2 * precision * recall / (precision + recall)
            f1_score[np.isnan(f1_score)] = 0
            label_disagreement = {k: v.round(3) for k, v in enumerate(FP / (FP + TN))}

            metrics = {
                "accuracy": accuracy,
                "f1_score_per_class": {k: v for (k, v) in enumerate(f1_score)},
                "f1_score": f1_score.mean(),
                "precision_per_class": {k: v for (k, v) in enumerate(precision)},
                "precision": precision.mean(),
                "recall_per_class": {k: v for (k, v) in enumerate(recall)},
                "recall": recall.mean(),
                "confusion_matrix": confusion_matrix.tolist(),
                "acc_per_class": acc_per_class,
                "label_disagreement": label_disagreement,
            }

        # save metrics to S3
        object_key = os.path.join(self.expt_dir, "metrics_{}.pkl".format(self.cur_loop))
        self.client.multi_part_upload_with_s3(
            metrics, self.bucket_name, object_key, "pickle"
        )
        return

    def infer(self, args):
        r"""
        A wrapper for your `infer` function which writes outputs to the specified S3 bucket. Returns `None`.

        Args:
           args: a dict whose keys include all of the arguments needed for your `infer` function which is defined in `processes.py`.

        """
        self.logger.info(
            "Getting insights on currently unused/unlabelled train data"
        )
        self.logger.warning(
            "This may take some time. Please be patient ............"
        )

        ts = range(self.meta_data["train_size"])
        self.unlabeled = sorted(list(set(ts) - set(self.labeled)))
        args['cur_loop'] = self.cur_loop
        outputs = self.infer_fn(
            args, unlabeled=deepcopy(self.unlabeled), ckpt_file=self.ckpt_file
        )

        if outputs is not None:
            outputs = outputs['outputs']
            # Remap to absolute indices
            remap_outputs = {}
            for _, (k, v) in enumerate(outputs.items()):
                ix = self.unlabeled.pop(0)
                remap_outputs[ix] = v

            pre_softmax = {}
            for k, v in remap_outputs.items():
                pre_softmax[k] = v['pre_softmax']

            self.logger.info(
                "Sending assessments on unlabelled train set to Alectio team"
            )

            if "Classification" in self.type:
                object_key = os.path.join(self.expt_dir, 'pre_softmax_{}.pkl'.format(self.cur_loop))
                self.client.multi_part_upload_with_s3(
                    pre_softmax,
                    self.bucket_name,
                    object_key,
                    "pickle"
                )

        else:

            if "Object Detection" in self.type:
                object_key = os.path.join(self.expt_dir, 'infer_outputs_{}.db'.format(self.cur_loop))
                local_db = os.path.join(self.args['EXPT_DIR'], 'infer_outputs_{}.db'.format(self.cur_loop))
                # upload sqlite db
                self.client.upload_file(local_db, self.bucket_name, object_key)
                # delete local version
                os.remove(local_db)
                
        return

    def __call__(self, debug=False, host="0.0.0.0", port=5000):
        r"""
        A wrapper for your `test` function which writes predictions and ground truth to the specified S3 bucket. Returns `None`.

        Args:
           debug (boolean, Default=False): If set to true, then the app runs in debug mode. See https://flask.palletsprojects.com/en/1.1.x/api/#flask.Flask.debug.
           host (str, Default='0.0.0.0'): the hostname to be listened to.
           port(int, Default:5000): the port of the webserver.

        """
        # As of now SDK will start the experiment everytime the SDK is initlized
        #TODO:: ADD RESUME OPTION

        # print("Starting the Experiment. Alectio is fetching indices, please wait.")
        self.backend = Backend(self.client_token)
        response = self.backend.startExperiment()
        print(response)
        count = 0
        if response == "Started":
            while True:
                response_child = self.backend.getSDKResponse()
                if response_child['status'] == "Fetched":
                    print('\n')
                    one_loop_response = self.one_loop(response_child)
                    print('\n')
                    print(one_loop_response)
                    count = 0
                if response_child['status'] == "Finished":
                    print('\n')
                    print("Experiment complete")
                    os.environ["AWS_ACCESS_KEY_ID"] = " "
                    os.environ["AWS_SECRET_ACCESS_KEY"] = " "
                    break
                if response_child['status'] == "Failed":
                    if count == 0:
                        print('\n')
                        print('Waiting for server.', end = '', flush=True)
                        count += 1
                        time.sleep(10)
                    else:
                        time.sleep(10)
                        print(".", end = '', flush=True)
