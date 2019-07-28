#!/usr/bin/python
import sys
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from src.logger.LOG import LOG
from src.AppConstants import AppConstants
from src.TVEKDNN.activators.ActivatorFunctions import TVEKActivatorFactory

"""
.. module:: CustomDNN
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Thomas Vimal Easo K<thomasvml@gmail.com>


"""

class CustomDNN():

	''' CustomDNN is a base class used as a reference for creating your custom DNN with ease
	'''
	def __init__(self):
		''' Default Constructor
		'''
		pass

	def generateAndInitData(self, vn_samples, vn_features):
		"""This is a instance method of CustomDNN used for generating data

		:param vn_samples: Number of row/samples for generated dataset.
		:type vn_samples: Int

		:param vn_features: Number of columns/features for generated dataset.
		:type vn_features: Int

		:return: True if generation is Success else False
		:rtype: bool
		"""
		tempX, tempy = make_classification(n_samples=vn_samples, n_features=vn_features,

								   n_informative=vn_features, n_redundant=0,

								   n_clusters_per_class=2, random_state=26)
		self.setData(tempX, tempy)
		LOG.D(" -> X:{} Y:{}".format(tempX.shape, tempy.shape))
		LOG.I("Dimension Generation - Successful")
		return True

	def setData(self, vX, vY):
		''' This is a instance method of CustomDNN used for initializing input dataset

        :param X: Number of row/samples.
        :type X: np.array

        :param Y: Number of columns/features.
        :type Y: np.array

        :return: True if generation is Success else False
        :rtype: bool
        '''
		self.X, self.y = vX, vY
		LOG.D(" -> X:{} Y:{}".format(self.X.shape, self.y.shape))
		LOG.I("Data Initialization - Successful")
		self.normalizeData()
		return True

	def visualizeData(self):
		''' This is a instance method of CustomDNN used for initializing input dataset

		:param None: None
		:type None: None

		:return: True if generation is Success else False
		:rtype: bool
		'''
		colors = ['black', 'yellow']
		cmap = ListedColormap(colors)
		plt.figure()
		plt.title('Data Visualization - Data Spread')
		plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, marker='o', s=50, cmap=cmap, alpha=0.5)
		filename = AppConstants.getRootPath() + 'analysis_reports/1.DataHeatMapVisualization.png'
		plt.savefig(filename, bbox_inches='tight')
		LOG.I("Visualization map is saved to {}".format(filename));
		return True

	def normalizeData(self):
		''' This is a instance method of CustomDNN used for normalizing input dataset

		:param None: None
		:type None: None

		:return: True if normalization is Success else False
		:rtype: bool
		'''
		col_max = np.max(self.X, axis=0)
		col_min = np.min(self.X, axis=0)
		self.X = np.divide(self.X - col_min, col_max - col_min)
		LOG.D(" -> X:{}".format(self.X.shape))
		LOG.I("Normalization is successful");
		return True

	def setActivationFunction(self, activationFunction):
		''' This is a instance method of CustomDNN used for initialize network parameters(weight & bias) for input dataset

		:param activationFunction: Activation function for Neuron
		:type activationFunction: String

		:return: True if parameter initialization is Success else False
		:rtype: bool
		'''
		bReturnValue = False
		self.activationFunction = activationFunction
		LOG.I("Activation Function({}) is configured successfully".format(self.activationFunction));
		return bReturnValue


	def prepareDataAndInitialize(self):
		''' This is a instance method of CustomDNN used for preparing input dataset by reordering its dimensions

        :param None: None
        :type None: None

        :return: True if generation is Success else False
        :rtype: bool
        '''
		# X(numberofsamples x numberoffeatures), Y(numberofsamples x 1)
		self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=25)
		LOG.D(" -> Train Test Split - shape of X_train:{} shape 0f Y_train:{}".format(self.X_train.shape, self.Y_train.shape))
		self.numOfSamples = self.X_train.shape[0]
		self.numOfFeatures = self.X_train.shape[1]
		LOG.D(" -> Number of samples:{} features:{}".format(self.numOfSamples, self.numOfFeatures))
		self.numberofNodesInLayerZero = 1
		self.numberofLevel = 1
		self.setParameters()
		if (self.numOfFeatures == self.W.T.shape[1]):
			self.X_train = self.X_train.T
			self.Y_train = self.Y_train.reshape(1, len(self.Y_train))
			self.X_test = self.X_test.T
			self.Y_test = self.Y_test.reshape(1, len(self.Y_test))

		LOG.D(" -> Data Reordering X_train:{} Y_train:{}".format(self.X_train.shape, self.Y_train.shape))
		LOG.I("Network is intialized successfully");
		# X(numberoffeatures x numberofsamples), Y(1 x numberofsamples) - for matrix operation
		LOG.D("Dimension Correction - shape of X_train:{} shape 0f Y_train:{} after transformation".format(self.X_train.shape, self.Y_train.shape))

	def setParameters(self):
		self.W = np.zeros((self.numOfFeatures, 1), dtype=int)
		self.b = 0
		LOG.D(" -> Weight:{}".format(self.W.shape))
		self.parameters = {"W": self.W, "b": self.b}

	def forwardPropagation(self):
		''' This is a instance method of CustomDNN used for forward propagating the neural net

		:param None: None
		:type None: None

		:return: True if generation is Success else False
		:rtype: bool
		'''
		self.W = self.parameters["W"]
		self.b = self.parameters["b"]
		self.Z = np.dot(self.W.T, self.X_train) + self.b
		self.A = self.applyActivator(self.dZ)
		LOG.D(" -> A:{}".format(self.A.shape))
		LOG.D("Forward Propogation is completed successfully");

		return True

	def cost(self):
		''' This is a instance method of CustomDNN used for calculating cost of DNN

		:param None: None
		:type None: None

		:return: True if generation is Success else False
		:rtype: bool
		'''
		return -1 * np.sum((self.Y_train * np.log(self.A)) + ((1 - self.Y_train) * (np.log(1 - self.A))))/ self.numOfSamples

	def backPropagration(self):
		''' This is a instance method of CustomDNN used for back propogation

		:param None: None
		:type None: None

		:return: True if generation is Success else False
		:rtype: bool
		'''
		dZ = self.A - self.Y_train
		LOG.D(" -> X_train:{} dZ.T:{}".format(self.X_train.shape, dZ.T.shape));
		self.dW = (np.dot(self.X_train, dZ.T)) / self.numOfSamples
		self.db = np.sum(dZ) / self.numOfSamples
		LOG.D(" -> dW:{} db:{}".format(self.dW, self.db));
		LOG.D("Back Propogation is completed successfully");
		return True

	def updateParameters(self, learning_rate):
		''' This is a instance method of CustomDNN used for updating tuning parameters

		:param learning_rate: (0,1] to mention the rate at which model should learn
		:type learning_rate: float

		:return: True if generation is Success else False
		:rtype: bool
		'''
		W = self.parameters["W"] - (learning_rate * self.dW)
		b = self.parameters["b"] - (learning_rate * self.db)
		self.parameters =  {"W": W, "b": b}
		return True

	def applyActivator(self, dZ):
		''' This is a instance method of CustomDNN used for updating tuning parameters

		:param dZ: Non Activated Array of Input
		:type dZ: np.array

		:return: Activated Array of outputs
		:rtype: np.array
		'''
		return TVEKActivatorFactory().MakeActivator(self.activationFunction, dZ)

	def model(self, num_iter, learning_rate):
		''' This is a instance method of CustomDNN used for creating the model

		:param learning_rate: (0,1] to mention the rate at which model should learn
		:type learning_rate: float

		:param num_iter: Number of times the model should be retrained to achieve more accuracy
		:type num_iter: int

		:return: True if generation is Success else False
		:rtype: bool
		'''
		self.prepareDataAndInitialize()
		for i in range(num_iter):
			self.forwardPropagation()
			if (i % 100 == 0):
				LOG.I("Cost after {} iteration: {}".format(i, self.cost()))
			self.backPropagration()
			parameters = self.updateParameters(learning_rate)
		return parameters


"""
Functionality includes:
1. Automatic Commenting support
2. Automatic testing support
"""