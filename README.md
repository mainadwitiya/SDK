# Requirements
* Python3 (Required)
* PIP3    (Required)
* Ubuntu 16.04+ / MacOS / Windows 10
* GCC / C++ (Will depend on OS you are using. Ubuntu, MacOS it comes default. Some falvours of linux distribution like Amazon Linux/RED Hat linux might not have GCC or C++ realted libraires installed)

For this tutorial, we are assuming you are using Python3 and PIP3. Also, make sure you have the necessary build tools installed (might vary from OS to OS). If you get any errors while installing any dependent packages feel free to reach out to us but most of it can quickly be solved by a simple Google search.  

# Alectio SDK

AlectioSDK is a package that enables developers to build an ML pipeline as a Flask app to interact with Alectio's
platform.
It is designed for Alectio's clients, who prefer to keep their model and data on their on server.

The package is currently under active development. More functionalities that aim to enhance robustness will be added soon, but for now the package provides a class `alectio_sdk.flask_wrapper.Pipeline` that inferfaces with customer-side
processes in a consistent manner. Customers need to implement 4 processes as python functions:

* A process to train the model
* A process to test the model
* A process to apply the model to infer on unlabeled data
* A process to assign each data point in the dataset to a unique index (Refer to one of the examples to know how)

### Train the Model
The logic for training the model should be implemented in this process. The function should look like:

```python
def train(payload):
    # get indices of the data to be trained
    labeled = payload['labeled']

    # get checkpoint to resume from
    resume_from = payload['resume_from']

    # get checkout to save for this loop
    ckpt_file = payload['ckpt_file']

    # implement your logic to train the model
    # with the selected data indexed by `labeled`
    return

```

The name of the function can be anything you like. It takes an argument `payload`, which is a
dictionary with 3 keys

| key | value |
| --- | ----- |
| resume_from | a string that specifies which checkpoint to resume from |
| ckpt_file | a string that specifies the name of checkpoint to be saved for the current loop |
| labeled | a list of indices of selected samples used to train the model in this loop |

Depending on your situation, the samples indicated in `labeled` might not be labeled (despite the variable
name). We call it `labeled` because in the active learning setting, this list represents the pool of
samples iteratively labeled by the human oracle.


### Test the Model
The logic for testing the model should be implemented in this process. The function representing this
process should look like:

```python
def test(payload):
    # the checkpoint to test
    ckpt_file = payload['ckpt_file']

    # implement your testing logic here


    # put the predictions and labels into
    # two dictionaries

    # lbs <- dictionary of indices of test data and their ground-truth

    # prd <- dictionary of indices of test data and their prediction

    return {'predictions': prd, 'labels': lbs}
```
The test function takes an argument `payload`, which is a dictionary with 1 key

| key | value |
| --- | ----- |
| ckpt_file | a string that specifies which checkpoint to test |

The test function needs to return a dictionary with two keys

| key | value |
| --- | ----- |
| predictions | a dictionary of an index and a prediction for each test sample|
| labels | a dictionary of an index and a ground truth label for each test sample|

The format of the values depends on the type of ML problem. Please refer to the [examples](./examples) directory for details.

## Apply Inference
The logic for applying the model to infer on the unlabeled data should be implemented in this process.
The function representing this process should look like:
```python
def infer(payload):
    # get the indices of unlabeled data
    unlabeled = payload['unlabeled']

    # get the checkpoint file to be used for applying inference
    ckpt_file = payload['ckpt_file']

    # implement your inference logic here


    # outputs <- save the output from the model on the unlabeled data as a dictionary
    return {'outputs': outputs}
```

The infer function takes an argument `payload`, which is a dictionary with 2 keys:

| key | value |
| --- | ----  |
| ckpt_file | a string that specifies which checkpoint to use to infer on the unlabeled data |
| unlabeled | a list of of indices of unlabeled data in the training set |


The `infer` function needs to return a dictionary with one key

| key | value |
| --- | ----- |
| outputs | a dictionary of indexes mapped to the models output before an activation function is applied |

For example, if it is a classification problem, return the output **before** applying softmax.
For more details about the format of the output, please refer to the [examples](./examples) directory.

## Installation
### 0. Key Management

If you have not already created your Client ID and Client Secret then do so by visiting:
1. open https://auth.alectio.com
2. Login there and click 'Create Client' Link, only change Name in the form and leave everything as it is
3. Click submit
4. Now you should have you Client ID and Client Secret
You will use them in the terminal where you are running alectio-kms

Install the Alectio Key Mangement Package by using:
```console
sudo pip install alectio-kms
```

During installation of alectio-kms make sure that the pip binary is also accessible for sudo/root user. That is use pip outside any of your virtual environment and pip binary is of python3 NOT python2.

Once the package is installed run:
```console
sudo alectio-kms
```

Upon running the package it will walk you through to get you keys setup
If you are running it for first time then it will ask you to enter your Client ID and Client Secret, it will open a web browser where you will authenticate yourself.
Upon successful authentication your Client ID, Client Secret and Auth Token will be save at /opt/alectio and will
also be outputted in your terminal. 

KMS will open the default browser (Most likely one that came default with your OS). Please make sure that you are already signed in with that browser otherwise, you will need to copy and paster the token retrieval URL from the terminal.

### 1. Set up a virtual environment
We recommend to set-up a virtual environment.

For example, you can use python's built-in virtual environment via:

```
python3 -m venv env
source env/bin/activate
```
### 2. Install AlectioSDK/requirements
```
pip install .
pip install -r requirements.txt
```
### 3. Configure aws credentials
We need to [configure the aws cli](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) by running
```
aws configure
```
Fill in your credentials as requested on your terminal
### 4. Run Examples

The remaining installation instructions are detailed in the [examples](./examples) directory. We cover one example for [topic classification](./examples/topic_classification), one example for [image classification](./examples/image_classification) and one example for [object detection](./examples/object_detection).