#!/usr/bin/python
import sys
from sklearn.datasets import make_classification
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
		return True

"""
Functionality includes:
1. Automatic Commenting support
2. Automatic testing support
"""