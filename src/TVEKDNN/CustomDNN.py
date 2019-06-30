#!/usr/bin/python
import sys
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.logger.LOG import LOG
from src.AppConstants import AppConstants

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
		self.X, self.y = make_classification(n_samples=vn_samples, n_features=vn_features,

								   n_informative=vn_features, n_redundant=0,

								   n_clusters_per_class=2, random_state=26)
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
		LOG.I("Data Initialization - Successful")
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
		plt.title('Non-linearly separable classes')
		plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, marker='o', s=50, cmap=cmap, alpha=0.5)
		plt.savefig(AppConstants.getRootPath() + 'analysis_reports/1.DataHeatMapVisualization.png', bbox_inches='tight')
		LOG.I("Visualization map is shown");
		return True

	def prepareData(self):
		''' This is a instance method of CustomDNN used for preparting input dataset by reordering its dimensions

        :param None: None
        :type None: None

        :return: True if generation is Success else False
        :rtype: bool
        '''
		self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=25)
		LOG.D("Train Test Split - shape of X_train:{} shape 0f Y_train:{}".format(self.X_train.shape, self.Y_train.shape))
		self.X_train = self.X_train.T
		self.Y_train = self.Y_train.reshape(1, len(self.Y_train))
		self.X_test = self.X_test.T
		self.Y_test = self.Y_test.reshape(1, len(self.Y_test))
		LOG.I("Dimension Correction - shape of X_train:{} shape 0f Y_train:{} after transformation".format(self.X_train.shape, self.Y_train.shape))

"""
Functionality includes:
1. Automatic Commenting support
2. Automatic testing support
"""