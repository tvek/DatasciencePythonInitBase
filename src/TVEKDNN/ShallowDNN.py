#!/usr/bin/python
from src.logger.LOG import LOG
from src.TVEKDNN.CustomDNN import CustomDNN
import numpy as np

"""
.. module:: CustomDNN
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Thomas Vimal Easo K<thomasvml@gmail.com>


"""

class ShallowDNN(CustomDNN):

	''' ShallowDNN is a sub class of CustomDNN used to create a shallo neural network(1 hidden layer)
	'''
	def __init__(self):
		''' Default Constructor
		'''
		pass

	def setParameters(self):
		for nCurrentLevel in range(self.numberofLevel):
			for nCurrentNode in range(self.numberofNodesInLayerZero):
				W = np.zeros((self.numOfFeatures, self.numberofNodesInLayerZero), dtype=int)
				b = np.zeros(self.numberofLevel)
				LOG.D(" -> Weight:{}".format(self.W1.shape))
				self.parameters = {"W1": self.W1, "b1": self.b1}

	def forwardPropagation(self):
		''' This is a instance method of ShallowDNN used for forward propagating the neural net

		:param None: None
		:type None: None

		:return: True if generation is Success else False
		:rtype: bool
		'''
		self.W1 = self.parameters["W1"]
		self.b1 = self.parameters["b1"]
		self.W2 = self.parameters["W2"]
		self.b3 = self.parameters["b2"]
		self.Z1 = np.dot(self.W1.T, self.X_train) + self.b1
		self.A1 = self.applyActivator(self.Z1)
		self.Z2 = np.dot(self.W2.T, self.A1) + self.b2
		self.A2 = self.applyActivator(self.Z2)
		LOG.D(" -> A:{}".format(self.A2.shape))
		LOG.D("Forward Propogation is completed successfully");

		return True

	def cost(self):
		''' This is a instance method of ShallowDNN used for calculating cost of DNN

		:param None: None
		:type None: None

		:return: True if generation is Success else False
		:rtype: bool
		'''
		return -1 * np.sum((self.Y_train * np.log(self.A)) + ((1 - self.Y_train) * (np.log(1 - self.A))))/ self.numOfSamples

	def backPropagration(self):
		''' This is a instance method of ShallowDNN used for back propogation

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
