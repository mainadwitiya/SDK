[![Medium][medium-shield]][medium-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Python][python-shield]][python-url]
[![Opensource][Opensource-shield]][Opensource-url]


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://alectio.com/">
    <img src="pics/alectiologo.png" alt="Logo" width="500" height="120">
  </a>

  <h2 align="center">Getting started with AlectioSDK</h2>

  <p align="center">
    <summary> <strong>Info Hub </strong></summary>
    <a href="https://medium.com/alectio/everything-you-wanted-to-know-about-alectio-ee70e105f7f">Who are we »</a>
    <br />
    <a href="https://docs.alectio.com/docs/setup-overview">Setup docs »</a>
    <br />
    <a href="https://github.com/alectio/AlectioExample">Alectio Examples »</a>
    <br />
    <a href="https://www.youtube.com/watch?v=Y7lnQYmrlBI&t=43s">How to create a Project »</a>
    <br />
    <a href="https://www.youtube.com/watch?v=nGavMSt2miM&t=109s">How to create an Experiment »</a>
    <br />
    <a href="https://www.youtube.com/watch?v=0iEMnRfLPcw">How to do Active learning »</a>
    <br />
    <a href="https://github.com/alectio/SDK/issues">Request a Feature »</a>
    <br />
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary> <strong>Table of Contents </strong></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#custom-usage">Custom Usage</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#required-implementations">Required Implementations</a></li>
        <li><a href="#experiment-reproduceability">Experiment reproduceability</a></li>
      </ul>
    </li>
    <li><a href="#open-issues">Open issues</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://demo.alectio.com/app/SSO/login)

**AlectioSDK** is an open source python package that provides end users the facility to perform Data curation for training Machine Learning models. The package is model/dataset/framework agnostic. It is simple , efficient and works on-prem so that your ML training can be carried out smoothly, securely and most importantly efficiently.

### Built With
The SDK is mainly built using the following
* [Python](https://www.python.org/)
* [GRPC](https://github.com/grpc/grpc)
* [PyTorch](https://pytorch.org/docs/stable/index.html)
* [Tensorflow](https://www.tensorflow.org/)


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
* Python3 (Required)
* PIP3    (Required)
* Ubuntu 16.04+ / MacOS / Windows 10
* GCC / C++ (Will depend on OS you are using. Ubuntu, MacOS it comes default. Some falvours of linux distribution like Amazon Linux/RED Hat linux might not have GCC or C++ realted libraries installed)


### Installation
Please follow instructions [here](https://docs.alectio.com/docs/setup-overview) to get started with the installation. If you want to give a quick trial run with our ready to go example, follow the instructions [here](https://github.com/alectio/cifar_image_classification)

<!-- USAGE EXAMPLES -->
## Custom Usage
Once you are done installing the above you are good to go. If you want to run one of our examples you should follow the remaining installation instructions detailed in the [examples](https://github.com/alectio/AlectioExamples) directory. We cover examples for [topic classification](https://github.com/alectio/AlectioExamples/topic_classification), for [image classification](https://github.com/alectio/AlectioExamples/image_classification) and for [object detection](https://github.com/alectio/AlectioExamples/object_detection).

### Required implementations 
To use AlectioSDK on a custom model/dataset you are required to create the following two python files inside your model implementation folder. Lets say for you have your model implemented in folder Detector  

&nbsp;&nbsp;.<br />
&nbsp;&nbsp;├── Detector <br />
&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── processes.py <br />
&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── main.py <br />
&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── config.yaml <br />
&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── < other model dependancy files/Folders > <br />

#### processes.py
AlectioSDK needs you to implement 4 main processes so that the SDK can be used to perform model and data optimizations using your own machine learning model and your data:

* Training process - A process to train the model 
* Testing process  - A process to test the model
* Infer process    - A process to apply the model to infer on unlabeled data
* Datamap process  - A process to assign each data point in the dataset to a unique index (Refer to one of the examples to know how)

##### Training process
The training logic to train your ML model needs to implemented here. The structure of the function should look like below
The logic for training the model should be implemented by you in this process. The function should look like below and should contain the following items:

```python
def train(args, labeled, resume_from, ckpt_file):
    """
    Get list of indices of the data to be trained in your dataloader.
    Note the SDK will pause for you to label these before you can train
    """
    labeledindices = labeled

    """
    Get checkpoint to resume from. Since the training is incremental 
    you may or maynot choose to resume from the previous checkpoint. 
    We suggest you not to resume from previous checkpoint to remove 
    previous model biases during training
    """
    resumeckpt = resume_from 
    
    """
    Use the name given by the SDK here to save your best model , this
    checkpoint is what will be used by other processes in this file
    """
    ckpt_file = ckpt_file

    # implement your logic to train the model
    # with the selected data indexed by `labeled`
    return

```
The name of the function can be anything you like. It takes the following arguments

| Arguments | Description |
| --- | ----- |
| args | a yaml file set up at your model folder to take model arguments|
| resume_from | a string that specifies which checkpoint to resume from, we suggest you to clear weights and not resume from previous loop checkpoint |
| ckpt_file | a string that specifies the name of checkpoint to be saved for the current loop, use this string to save your best checkpoint using your logic on what you consider as the best checkpoint |
| labeled | a list of indices of selected samples used to train the model in this loop |

Depending on your situation, the samples indicated in `labeled` might not be labeled (despite the variable
name). We call it `labeled` because in the active learning setting, this list represents the pool of
samples iteratively labeled by the human oracle. Make sure you label these data points when the SDK pauses after each loop. 

##### Testing process
The logic for testing the model should be implemented in this process. The function representing this
process should look like:

```python
def test(args, ckpt_file):
    # the checkpoint to test
    ckpt_file = ckpt_file

    # implement your testing logic here


    # put the predictions and labels into
    # two dictionaries

    # lbs <- dictionary of indices of test data and their ground-truth

    # prd <- dictionary of indices of test data and their prediction

    return {'predictions': prd, 'labels': lbs}
```
The test function takes 2 arguments as below 

| Arguments | Description |
| --- | ----- |
| args | a yaml file set up at your model folder to take model arguments|
| ckpt_file | a string that specifies checkpoint that should be used for testing|

The test function needs to return a dictionary with two keys

| key | value |
| --- | ----- |
| predictions | a dictionary of an index and a prediction for each test sample|
| labels | a dictionary of an index and a ground truth label for each test sample|

The format of the values depends on the type of ML problem. Please refer to the [examples](https://github.com/alectio/AlectioSDK/tree/master/examples) directory for details regarding what is needed for each usecase.

##### Infer process 
The logic for applying the model to infer on the unlabeled data should be implemented in this process.
The function representing this process should look like:
```python
def infer(args, unlabeled,ckpt_file):
    """
    Get the list of indices of unlabeled data. Use these indices in your 
    dataloader to infer on unlabeled pool of data
    """
    unlabeledindices = unlabeled

    # get the checkpoint file to be used for applying inference
    ckpt_file = ckpt_file

    # implement your inference logic here
    
    """
    outputs <- save the output from the model on the unlabeled data 
    as a dictionary
    """
    return {'outputs': outputs}
```

The infer function takes an argument the following 2 arguments

| key | value |
| --- | ----  |
| ckpt_file | a string that specifies which checkpoint to use to infer on the unlabeled data |
| unlabeled | a list of of indices of unlabeled data in the training set |

The `infer` function needs to return a dictionary with one key

| key | value |
| --- | ----- |
| outputs | a dictionary of indexes mapped to the models output before an activation function is applied |

For example, if it is a classification problem, return the output **before** applying softmax.
For more details about the format of the output, please refer to the [examples](https://github.com/alectio/AlectioSDK/tree/master/examples) directory.

##### Datamap process
The logic that creates a reference table to refer to all of your data
The function representing this process should look like:

```python
def getdatasetstate(args):
    """
    Create a loader object with 100 % of your data i.e all of labelled, unlabelled 
    data that you are planning to use for this experiment
    """
    loader = DataLoader(datasetobject)     #Note this can be done in any framework that you choose to use , this is just an example
    trainreference = {}
    for ix, pathtorecord, _ in loader:
        trainreference[ix] = pathtorecord

    return trainreference
```

The `getdatasetstate` function needs to return a dictionary which contains index mapped to path of the record 

#### main.py
Use the main.py to feed in all 4 above processes into AlectioSDK
Refer to one of our [examples](https://github.com/alectio/AlectioSDK/tree/master/examples) and it will be easier for you to mimic the same. All you need now is an experiment name that you can come up with and unique experiment token that you can get from Alectio's FE

```python

#Sample format
#Import necessary modules

from alectio_sdk.sdk import Pipeline
from processes import train, test, infer, getdatasetstate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config','-c',
    type= str,
    required =True,
    default = './config.yaml',
    help = 'Config to use to trigger AlectioSDK')
    params, _ = parser.parse_known_args()
    args = yaml.safe_load(open(params.config,'r'))
    
    AlectioPipeline = Pipeline(
    name=args["exp_name"],                  # your experiment name , can be anything of type str
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    args=args,
    token="<alectio-experiment-token>",   # your unique token of type str
    )
    AlectioPipeline()




```


#### config.yaml (optional)
Put in all the requirements that are required for the model to train. This will be read and used in processes.py when the model trains. For example if config.yaml looks like this 

``` python
exptname:  "ManualAL"         
# Model configs
backbone:     "Resnet101"
description:  "Pedestrian detection"
...
..

```
you can access them inside your any of the above 4 processes as lets say args["backbone"] , args["description"] etc



### Experiment reproduceability
AlectioSDK is just an orchestrator to orchestrate your training process on-prem. Please be adviced that randomness has a significant impact on your results. We have included utilities that you can use to reduce randomness significantly. But in some cases like [Pytorch](https://pytorch.org/docs/stable/notes/randomness.html) randomness in certain network parameters still exists. We advice you to use the following guidelines to reduce this as much as possible if you want to see reproduceable results

#### Pytorch
To ensure reproduceability in Pytorch use the following

* set num_workers = 0 in your dataloader 
* Inside your processes.py file ensure the following things exist (Note: This may make your training slow, we trust you to assess the tradeoff between speed and reproduceability. Using distributed training can also induce randomness in your training thereby affecting the end result of your experiments for the better or the worst)

```python
from alectio_sdk.torch_utils.utils import setpytorchreproduceability

def train(args, labeled, resume_from, ckpt_file):

    # Ensuring reproduceability
    setpytorchreproduceability(seed= 42)
    """
    Get list of indices of the data to be trained in your dataloader.
    Note the SDK will pause for you to label these before you can train
    """
    labeledindices = labeled

    """
    Get checkpoint to resume from. Since the training is incremental 
    you may or maynot choose to resume from the previous checkpoint. 
    We suggest you not to resume from previous checkpoint to remove 
    previous model biases during training
    """
    resumeckpt = resume_from 
    
    """
    Use the name given by the SDK here to save your best model , this
    checkpoint is what will be used by other processes in this file
    """
    ckpt_file = ckpt_file

    # implement your logic to train the model
    # with the selected data indexed by `labeled`
    return

```

#### Tensorflow
To ensure reproduceability in Tensorflow use the following

* Inside your processes.py file ensure the following things exist (Note: This may make your training slow, we trust you to assess the tradeoff between speed and reproduceability. Using distributed training can also induce randomness in your training thereby affecting the end result of your experiments for the better or the worst)

```python
from alectio_sdk.tensorflow_utils.utils import settfreproduceability

def train(args, labeled, resume_from, ckpt_file):

    # Ensuring reproduceability
    settfreproduceability(seed= 42)
    """
    Get list of indices of the data to be trained in your dataloader.
    Note the SDK will pause for you to label these before you can train
    """
    labeledindices = labeled

    """
    Get checkpoint to resume from. Since the training is incremental 
    you may or maynot choose to resume from the previous checkpoint. 
    We suggest you not to resume from previous checkpoint to remove 
    previous model biases during training
    """
    resumeckpt = resume_from 
    
    """
    Use the name given by the SDK here to save your best model , this
    checkpoint is what will be used by other processes in this file
    """
    ckpt_file = ckpt_file

    # implement your logic to train the model
    # with the selected data indexed by `labeled`
    return

```

<!-- ROADMAP -->
## Open issues

See the [open issues](https://github.com/alectio/SDK/issues) for a list of known issues.



<!-- CONTRIBUTING -->
## Contributing

Contributions are **greatly appreciated**.

<!-- CONTACT -->
## Contact

Twitter - [@alectiolessdata](https://twitter.com/alectiolessdata) 
<br />
Email - info@alectio.com
<br />

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* Almost all of our [Examples](https://github.com/alectio/AlectioSDK/tree/master/examples) are open sourced well known models. We don't claim ownership for any of those.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[medium-shield]: https://img.shields.io/badge/M-Medium-blue
[medium-url]: https://medium.com/alectio
[python-shield]: https://img.shields.io/badge/Made%20with-Python-blue
[python-url]: https://www.python.org/
[linkedin-shield]: https://img.shields.io/badge/in-Linkedin-blue
[linkedin-url]: https://www.linkedin.com/company/alectio/
[Opensource-shield]: https://img.shields.io/badge/open-source-blue
[Opensource-url]: https://github.com/alectio/SDK/pulls
[product-screenshot]: pics/screenshot.png

