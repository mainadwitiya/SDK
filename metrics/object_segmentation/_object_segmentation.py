# -*- coding: utf-8 -*-
"""
@author: arun
"""
import logging
from .SegCM import SegCM as CM
from collections import OrderedDict
from .rangeutils import SemLaserScan     #####  Requirement for rangenet implementation Source: http://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf



class SegMetrics(object):
  """
  Segmentation Evaluation 
  Current Support :
  2D
  3D - Support only Rangenet as seen on http://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf

  """


  def __init__(self,
               n_classes,
               labels,
               return_2D = True ,
               return_3D = True , 
               rangenet = True , 
               default_ranges = None , 
               include_ranges = True, 
               range_labels = OrderedDict()
               ):
    
    self.n_classes = n_classes
    self.labels = labels

    if rangenet:
      logging.info("Initializing Segmentation evaluation for {} classes with range projection from Rangenet " . format(n_classes))
      if include_ranges is True and default_ranges is None:
        raise TypeError("Default ranges cannot be None when include_ranges is set to True")
      self.default_ranges = default_ranges
      self.include_ranges = include_ranges
    else:
      logging.info("Initializing Segmentation evaluation for {} classes " . format(n_classes))


  def num_classes(self):
    return self.n_classes

  def compute(self):
    self.max_range_conf.add_single_comparison(gt_scan.sem_label,
                                                  pred_scan.sem_label)


  def _reset3D(self):
    logging.info("Resetting metrics evaluation for segmentation")
    self.pclCM = CM(nClasses=nclasses, labels=self.labels, task="Segmentation")   #3D
    if include_ranges:
      self.pclrangeCM = OrderedDict((range_spec, CM(nClasses = nclasses, labels = self.labels , task="Segmentation")) for range_spec in self.default_ranges)
      self.rangemiou = OrderedDict((range_spec, {}) for range_spec in self.default_ranges)
      self.rangeiou = OrderedDict((range_spec, {}) for range_spec in self.default_ranges)
      self.rangedice = OrderedDict((range_spec, {}) for range_spec in self.default_ranges)
      self.rangemdice = OrderedDict((range_spec, {}) for range_spec in self.default_ranges)
      self.rangeacc = OrderedDict((range_spec, {}) for range_spec in self.default_ranges)
      self.rangemacc = OrderedDict((range_spec, {}) for range_spec in self.default_ranges)
    else:
      self.pclrangeCM = {}


  def _reset2D(self):
    logging.info("Resetting metrics evaluation for segmentation")
    self.imageCM = CM(nClasses=nclasses, labels=self.labels, task="Segmentation") #2D
    

  def evaluate3D(self,gt, preds , rangelabels = OrderedDict()):
    """
    3D segmentation evaluyations only includes support for Rangenet types

    """
    logging.info("Resetting metrics evaluation for 3D rangenet based segmentation")
    self._reset3D()

    # Evaluating range wise and non rangewise confusion matrix
    for k in gt.keys():
      for groundtruth, predicted in zip(gt[k], pred[k]):
        self.pclCM.updateCM(groundtruth, predicted)
      if self.include_ranges:
        if not rangelabels:
          raise ValueError("Range labels cannot be empty for range based evaluations , OrderedDict in format [GTlabels, predlabels] is required")
        for rangeval, rangeCM in self.pclrangeCM.items():
          rangeGT , rangepreds = rangelabels[k][rangeval]
          self.pclrangeCM[k][rangeval].updateCM(rangeGT, rangepreds)

  def evaluate2D(self, gt, preds):
    """
    3D segmentation evaluyations only includes support for Rangenet types

    """
    logging.info("Resetting metrics evaluation for 2D Segmentation")
    self._reset2D()
    for k in gt.keys():
      for groundtruth, predicted in zip(gt[k], pred[k]):
        self.imageCM.updateCM(groundtruth, predicted)


  def get3DrangeCM(self):
    if self.include_ranges:
      return self.pclrangeCM
    else:
      return None

  def get3DCM(self):
    return self.pclCM


  def get3Drangeacc(self):
    for range_spec in self.self.rangeacc.keys():
      self.self.rangeacc[range_spec] = self.calcaccuracy(self, self.pclrangeCM[range_spec])
    return self.rangeacc
  
  def get3Dacc(self):
    return self.calcaccuracy(self, self.pclCM)
  

  def get3Dmacc(self):
    return self.calcmaccuracy(self, self.pclCM)    

  def get3Drangemacc(self):
    for range_spec in self.rangemacc.keys():
      self.rangemacc[range_spec] = self.calcmaccuracy(self, self.pclrangeCM[range_spec])
    return self.rangemacc


  def get3DrangeIOU(self):
    for range_spec in self.rangeiou.keys():
      self.rangeiou[range_spec] = self.calcIOU(self, self.pclrangeCM[range_spec])
    return self.rangeiou
  
  def get3DIOU(self):
    return self.calcIOU(self, self.pclCM)
  

  def get3DmIOU(self):
    return self.calcMIOU(self, self.pclCM)    

  def get3DrangemIOU(self):
    for range_spec in self.rangemiou.keys():
      self.rangemiou[range_spec] = self.calcMIOU(self, self.pclrangeCM[range_spec])
    return self.rangemiou

  def get3DDICE(self):
    return self.calcDICE(self, self.pclCM)

  def get3DrangeDICE(self):
    for range_spec in self.rangedice.keys():
      self.rangedice[range_spec] = self.calcDICE(self, self.pclrangeCM[range_spec])
    return self.rangedice
  
  def get3DmDICE(self):
    return self.calcmDICE(self, self.pclCM)    

  def get3DrangemDICE(self):
    for range_spec in self.rangemdice.keys():
      self.rangemdice[range_spec] = self.calcmDICE(self, self.pclrangeCM[range_spec])
    return self.rangemdice




  def get2DCM(self):
    return self.imageCM

  def get2DAcc(self):
    return self.calcaccuracy(self, self.imageCM)

  def get2DmAcc(self):
    return self.calcmaccuracy(self, self.imageCM)

  def get2DIOU(self):
    return self.calcIOU(self, self.imageCM)

  def get2DmIOU(self):
    return self.calcMIOU(self, self.imageCM)


  def get2DDICE(self):
    return self.calcDICE(self, self.imageCM)

  def get2DmDICE(self):
    return self.calcmDICE(self, self.imageCM)

  def get2DfwIOU(self):
    return self.calcfwIOU(self, self.imageCM)

  
  def calcfwIOU(self, inputCM):
    intersection = np.diagonal(inputCM)
    #Calculate area of Union , Union = 
    union = (np.sum(inputCM, axis=0)+np.sum(inputCM, axis=1)) - intersection
    classwiseiou = (intersection/union)
    return np.divide(np.sum(np.multiply(np.sum(inputCM, axis=0),classwiseiou)),np.sum(inputCM))


  def calcIOU(self, inputCM):
    """
    Calculates and returns classwise IOU for the input segmentation labels and predictions


    IOU  = (Area of Intersection)/(Area of union)
    """
    
    #Calculate area of Intersection 
    intersection = np.diagonal(inputCM)
    #Calculate area of Union , Union = 
    union = (np.sum(inputCM, axis=0)+np.sum(inputCM, axis=1)) - intersection
    classwiseiou = (intersection/union)
    classwiseioudict = {}

    for c in range(self.n_classes):
      classwiseioudict[c] = classwiseiou[c]

    return classwiseioudict


  def calcMIOU(self, inputCM):
    """
    Calculates and returns IOU for the input segmentation labels and predictions
    """
    #Calculate area of Intersection 

    intersection = np.diagonal(inputCM)
    #Calculate area of Union , 
    union = (np.sum(inputCM, axis=0)+np.sum(inputCM, axis=1)) - intersection

    return np.nanmean(intersection/union)


  def calcDICE(self, inputCM):
    #Calculate area of Intersection 
    intersection = np.diagonal(inputCM)

    #Calculate total pixels combined
    totalpixels = (np.sum(inputCM, axis=0)+np.sum(inputCM, axis=1))

    classwiseDICE = ((2*intersection)/totalpixels)
    classwiseDicedict = {}

    for c in range(self.n_classes):
      classwiseDicedict[c] = classwiseDICE[c]

    return classwiseDicedict

  def calcmDICE(self, inputCM):
    #Calculate area of Intersection 
    intersection = np.diagonal(inputCM)

    #Calculate total pixels combined
    totalpixels = (np.sum(inputCM, axis=0)+np.sum(inputCM, axis=1))

    return np.nanmean((2*intersection)/totalpixels)


  def calcaccuracy(self, inputCM):
    #Calculate area of Intersection 
    intersection = np.diagonal(inputCM)

    #Calculate total pixels combined
    totalGTpixels = (np.sum(inputCM, axis=1))

    classwiseaccuracy = (intersection/totalGTpixels)
    classwiseaccdict = {}

    for c in range(self.n_classes):
      classwiseaccdict[c] = classwiseaccuracy[c]

    return classwiseaccdict

  def calcmaccuracy(self, inputCM):
    #Calculate area of Intersection 
    intersection = np.diagonal(inputCM)

    #Calculate total pixels combined
    totalGTpixels = (np.sum(inputCM, axis=1))

    return np.nanmean((intersection/totalGTpixels))

