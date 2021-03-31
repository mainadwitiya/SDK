from flask import jsonify
from flask import Flask, Response
from flask import request
from flask import send_file

# from waitress import serve
import numpy as np
import json
import requests
import os
import psutil
import time
import boto3
import logging
from copy import deepcopy
from .s3_client import S3Client
from alectio_sdk.metrics.object_detection import Metrics, batch_to_numpy
from alectio_sdk.metrics.classification import ClassificationMetrics
from alectio_sdk.metrics.multilabel_classification import MultiLabelClassificationMetrics
from alectio_sdk.metrics.object_segmentation import SegMetrics
from alectio_sdk.sdk.sql_client import create_connection
from alectio_sdk.backend.backend import Backend

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
        sentry_sdk.init(
            dsn="https://4eedcc29fa7844828397dca4afc2db32@o409542.ingest.sentry.io/5282336",
            integrations=[FlaskIntegration()]
        )
        """
        self.app = Flask(name)
        self.gunicorn_logger = logging.getLogger("gunicorn.error")
        self.app.logger.handlers = self.gunicorn_logger.handlers
        self.app.logger.setLevel(self.gunicorn_logger.level)
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
        self.labeled = []

    def _notifyserverstatus(self, logdir):
        logging.basicConfig(
            filename=os.path.join(logdir, "Appstatus.log"), level=logging.INFO
        )
        self.app.logger.info("Flask app from Alectio initialized successfully")
        self.app.logger.info(
            "Training checkpoints and other logs for current experiment will be written into the folder {}".format(
                logdir
            )
        )
        self.app.logger.info(
            "Press CTRL + C to exit flask app , if Flask app terminates in the middle use fuser -k <port number>/tcp to terminate current process and relaunch alectio sdk"
        )

    def _checkdirs(self, dir_):
        if not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)

    def check_model_activations(self, model_outputs):
        supported_activations = {'pre_softmax', 'softmax', 'sigmoid', 'logits'}
        activation_set = set()
        for k, v in model_outputs.items():
            if isinstance(v, dict):
                activation_set.update(set(v.keys()).intersection(supported_activations))
            if len(activation_set) == 0:
                raise TypeError('Incorrect logit output format encountered')

        return activation_set

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
        self.app.logger.info("Estimated time left for the experiment: {}".format(time_left))
        return time_left

    def one_loop(self, request):
        # Get payload args

        self.app.logger.info("Extracting payload arguments from Alectio")
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

        self.logdir = payload["experiment_id"]
        self._checkdirs(self.logdir)

        self._notifyserverstatus(self.logdir)
        self.app.logger.info("Valid payload arguments extracted")
        self.app.logger.info("Initializing process to train and optimize your model")
        returned_payload = self._one_loop(payload, self.args)
        self.app.logger.info("Optimization process complete !")
        self.app.logger.info(
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
            self.app.logger.info(
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
        self.app.logger.info("Extracting essential experiment params")

        # read selected indices upto this loop
        payload["cur_loop"] = int(payload["cur_loop"])
        # self.curout_loop = payload["cur_loop"]

        # Leave cur_loop - 1 when doing a 1 dag solution, when doing 2 dag cur_loop remains the same
        self.cur_loop = payload["cur_loop"]
        self.bucket_name = payload["bucket_name"]
        self.n_loop = payload["n_loop"]

        # type of the ML problem

        self.type = payload["type"]
        self.type = self.type.lower()
        print('Task type: {}'.format(self.type))
        # dir for expt log in S3
        expt_dir = [payload["user_id"], payload["project_id"], payload["experiment_id"]]

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

        self.app.logger.info("Essential experiment params have been extracted")
        # get meta-data of the data set
        self.app.logger.info("Verifying the meta.json that was uploaded by the user")
        key = os.path.join(self.project_dir, "meta.json")
        bucket = boto3.resource("s3").Bucket(self.bucket_name)

        json_load_s3 = lambda f: json.load(bucket.Object(key=f).get()["Body"])
        self.meta_data = json_load_s3(key)
        self.app.logger.info(
            "SDK Retrieved file: {} from bucket : {}".format(key, self.bucket_name)
        )

        # self.meta_data = self.client.read(self.bucket_name, key, "json")
        # logging.info('SDK Retrieved file: {} from bucket : {}'.format(key, self.bucket_name))

        self.labeled = []
        self.app.logger.info("Reading indices to train on")

        for i in range(self.cur_loop + 1):
            object_key = os.path.join(self.expt_dir, "selected_indices_{}.pkl".format(i))
            selected_indices = self.client.read(self.bucket_name, object_key=object_key, file_format="pickle")
            self.labeled.extend(selected_indices)


        if self.cur_loop == 0:

            self.resume_from = None
            self.app.logger.info(
                "Extracting indices for our reference, this may take time ... Please be patient"
            )
            self.state_json = self.getstate_fn(args)
            object_key = os.path.join(self.expt_dir, "data_map.pkl")
            self.app.logger.info("Extraction complete !!!")
            self.app.logger.info(
                "Creating index to records data reference for the current experiment"
            )
            self.client.multi_part_upload_with_s3(
                self.state_json, self.bucket_name, object_key, "pickle"
            )
            self.app.logger.info("Reference creation complete")

        else:

            # check if ckpt cur_loop - 1 exists, otherwise we need to download it from S3
            if not os.path.isfile(
                    os.path.join(self.args["EXPT_DIR"], f"ckpt_{self.cur_loop - 1}")
            ):
                # need to download the checkpoint files from S3
                self.app.logger.info(
                    "Starting to copy checkpoints for cloned experiment..."
                )
                self.client.download_checkpoints(
                    payload["bucket_name"],
                    payload["project_id"],
                    payload["experiment_id"],
                    payload["cur_loop"],
                    self.args["EXPT_DIR"],
                )
                self.app.logger.info(
                    "Finished downloading checkpoints for cloned experiment"
                )

            self.app.logger.info("Resuming from a checkpoint from a previous loop ")
            # two dag approach needs to refer to the previous checkpoint
            self.resume_from = "ckpt_{}".format(self.cur_loop - 1)

        self.ckpt_file = "ckpt_{}".format(self.cur_loop)
        self.app.logger.info("Initializing training of your model")

        self.train(args)
        self.app.logger.info("Training complete !")
        self.app.logger.info("Initializing testing of your model !")
        self.test(args)
        self.app.logger.info("Testing complete !")
        self.app.logger.info("Assessing current best model")
        self.infer(args)
        self.app.logger.info("Assesment complete ")
        self.app.logger.info(
            "Time to check what records to train on next loop , visit our front end for more details"
        )

        # Drop unwanted payload values
        del payload["type"]
        del payload["cur_loop"]
        del payload["bucket_name"]
        return payload

    def train(self, args):
        r"""
        A wrapper for your `train` function. Returns `None`.

        Args:
           args: a dict whose keys include all of the arguments needed for your `train` function which is defined in `processes.py`.

        """
        start = time.time()

        if self.labeled is not None and len(self.labeled) == 0:
            raise ValueError('Labeled indices from backend are empty or None')

        self.app.logger.info("Labeled records are ready to be trained")
        self.labeled.sort()  # Maintain increasing order

        if len(self.labeled) > len(set(self.labeled)):
            raise ValueError("There exist repeated records.")

        train_op = self.train_fn(
            args,
            labeled=deepcopy(self.labeled),
            resume_from=self.resume_from,
            ckpt_file=self.ckpt_file,
        )

        if train_op is not None:
            labels = train_op["labels"]
            unique, counts = np.unique(labels, return_counts=True)
            num_queried_per_class = {u: c for u, c in zip(unique, counts)}

        end = time.time()

        # @TODO compute insights from labels
        if train_op is not None:
            insights = {"train_time": end - start, "num_queried_per_class": num_queried_per_class}
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
        self.app.logger.info("Extracting test results ")
        res = self.test_fn(args, ckpt_file=self.ckpt_file)

        if self.type == "3d_segmentation":
            if "rangebased" in res:
                _3D_predictions, _3D_labels, _2D_predictions, _2D_labels, range_filtered, range_default = \
                    res["3Dpredictions"], res["3Dlabels"], res["2Dpredictions"], res["2Dlabels"], \
                    res["rangebased"], res["default"]
            else:
                _3D_predictions, _3D_labels, _2D_predictions, _2D_labels, range_filtered, range_default = \
                    res["3Dpredictions"], res["3Dlabels"], res["2Dpredictions"], res["2Dlabels"], None, None
        else:
            predictions, ground_truth = res["predictions"], res["labels"]
        self.app.logger.info("Writing test results to S3")

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
                self.expt_dir, "test_ground_truth.pkl".format(self.cur_loop)
            )
            self.client.multi_part_upload_with_s3(
                ground_truth, self.bucket_name, object_key, "pickle"
            )

        if "3D" in self.type:
            self.compute3D_metrics(_3D_predictions, _3D_labels, _2D_predictions, _2D_labels, range_filtered)
        else:
            self.compute_metrics(predictions, ground_truth)
        return

    def compute3D_metrics(self, pclpredictions, pcllabels, imgpredictions, imglabels, range_filtered, range_default):
        metrics = {}
        if self.type == "3d_object_detection":
            raise NotImplementedError("3D object detection evaluation has not been implemented yet")

        elif self.type == "3d_segmentation":
            m = SegMetrics(
                n_classes=len(self.meta_data["class_labels"]),
                labels=self.meta_data["class_labels"],
                return_2D=True,
                return_3D=True,
                rangenet=True,
                default_ranges=range_default,
                include_ranges=True,
                range_labels=range_filtered,
            )

            m.evaluate3D(pcllabels, pclpredictions, rangelabels=range_filtered)
            m.evaluate2D(imglabels, imgpredictions)
            metrics = {
                "3DCM": m.get3DCM(),
                "3DrangeCM": m.get3DrangeCM(),
                "3DmIOU": m.get3DmIOU(),
                "3DrangemIOU": m.get3DrangemIOU(),
                "3DIOU": m.get3DIOU(),
                "3Drange3DIOU": m.get3DrangeIOU(),
                "3DDICE": m.get3DDICE(),
                "3DmDICE": m.get3DmDICE(),
                "3DrangeDICE": m.get3DrangeDICE(),
                "3DrangemDICE": m.get3DrangemDICE(),
                "3DAcc": m.get3Dacc(),
                "3DmAcc": m.get3Dmacc(),
                "3DrangeAcc": m.get3Drangeacc(),
                "3DrangemAcc": m.get3Drangemacc(),
                "class_labels": self.meta_data["class_labels"],
            }

    def compute_metrics(self, predictions, ground_truth):
        metrics = {}
        if self.type == "2d_object_detection":
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

        elif self.type == "text_classification" or self.type == "image_classification":

            m = ClassificationMetrics(predictions, ground_truth)

            metrics = {
                "accuracy": m.get_accuracy(),
                "f1_score_per_class": m.get_f1_score_per_class(),
                "f1_score": m.get_f1_score(),
                "precision_per_class": m.get_precision_per_class(),
                "precision": m.get_precision(),
                "recall_per_class": m.get_recall_per_class(),
                "recall": m.get_recall(),
                "confusion_matrix": m.get_confusion_matrix(),
                "acc_per_class": m.get_acc_per_class(),
                "label_disagreement": m.get_label_disagreement(),
            }

        elif self.type == "2d_segmentation":

            m = SegMetrics(
                n_classes=len(self.meta_data["class_labels"]),
                labels=self.meta_data["class_labels"],
                return_2D=True,
                return_3D=False,
                rangenet=False,
                default_ranges=None,
                include_ranges=False,
                range_labels=None,
            )
            m.evaluate2D(ground_truth, predictions)

            metrics = {
                'confusion_matrix': m.get2DCM(),
                'pixel_acc': m.get2DmAcc(),
                'classwise_pixel_acc ': m.get2DAcc(),
                'freqw_iou': m.get2DfwIOU(),
                'mean_iou': m.get2DmIOU(),
                'classwise_iou': m.get2DIOU(),
                'classwise_dice': m.get2DDICE()}

        elif self.type == "multilabel_text_classification" or self.type == "multi_label_text_classification":

            m = MultiLabelClassificationMetrics(predictions, ground_truth)

            metrics = {
                "accuracy": m.get_accuracy(),
                "micro_f1": m.get_f1_score_micro(),
                "macro_f1": m.get_f1_score_macro(),
                "micro_precision": m.get_precision_micro(),
                "macro_precision": m.get_precision_macro(),
                "micro_recall": m.get_recall_micro(),
                "macro_recall": m.get_recall_macro(),
                "multilabel_confusion_matrix": m.get_confusion_matrix(),
                "hamming_loss": m.get_hamming_loss()
            }

        else:
            raise ValueError("The selected task type is currently not supported, received type : {}".format(self.type))

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
        self.app.logger.info(
            "Getting insights on currently unused/unlabelled train data"
        )
        self.app.logger.warning(
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
            # finds correct activation as key i.e. 'sigmoid' or 'logits'
            activation = self.check_model_activations(outputs)

            # Remap to absolute indices
            logits_dict = {}
            remap_dict = {}
            for key in outputs.keys():
                remap_dict[key] = self.unlabeled.pop(0)

            while len(outputs) > 0:
                for orig_ix, correct_ix in remap_dict.items():
                    val = outputs.pop(orig_ix)
                    logits_dict[correct_ix] = {}
                    for a in activation:
                        logits_dict[correct_ix][a] = val[a]

            self.app.logger.info(
                "Sending assessments on unlabelled train set to Alectio team"
            )

            if "classification" in self.type:
                object_key = os.path.join(self.expt_dir, 'logits_{}.pkl'.format(self.cur_loop))
                self.client.multi_part_upload_with_s3(
                    logits_dict,
                    self.bucket_name,
                    object_key,
                    "pickle"
                )
            elif "object_detection" or "semantic" in self.type:
                object_key = os.path.join(self.expt_dir, 'infer_outputs_{}.db'.format(self.cur_loop))
                local_db = os.path.join(self.args['EXPT_DIR'], 'infer_outputs_{}.db'.format(self.cur_loop))
                logits_conn = create_connection(local_db)

                if logits_conn is not None:
                    logits_cur = logits_conn.cursor()
                    sql = """
                            UPDATE indexes
                            SET row_id = ?
                            WHERE id = ?
                    """
                    for index, correct_ix in enumerate(self.unlabeled):
                        logits_cur.execute(sql, (correct_ix, index))

                    logits_conn.close()
                else:
                    raise ConnectionError('Cannot connect to softmax database')

                # upload sqlite db
                self.client.upload_file(local_db, self.bucket_name, object_key)
                # delete local version
                os.remove(local_db)
            else:
                raise ValueError('Use case not found')
        else:
            raise ValueError('Infer outputs not returned. You must return outputs from the implemented infer() '
                             'function. Please refer to the README on the Alectio SDK repository.')

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
        # TODO:: ADD RESUME OPTION

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
                        print('Waiting for server.', end='', flush=True)
                        count += 1
                        time.sleep(10)
                    else:
                        time.sleep(10)
                        print(".", end='', flush=True)

    @staticmethod
    def shutdown_server():
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            raise RuntimeError("Not running with the Werkzeug Server")
        func()

    @staticmethod
    def end_exp():
        print()
        print("======== Experiment Ended ========")
        print("Server shutting down...")
        p = psutil.Process(os.getpid())
        p.terminate()
        return "Experiment complete"