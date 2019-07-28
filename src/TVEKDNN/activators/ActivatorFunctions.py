import numpy as np

"""
.. module:: TVEKActivators
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Thomas Vimal Easo K<thomasvml@gmail.com>


"""

class TVEKActivatorFactory():

    def MakeActivator(self, type, input):
        activator = self.passactivator
        if(type == "sigmoid"):
            activator = self.sigmoid
        else:
            activator = self.passactivator

        output = activator(input)
        return output

    def passactivator(self, z):
        """This is a static method of TVEKActivators used for pass activator definition

        :param z: Output of the neuron before activation
        :type z: np.array

        :return: Output of neuron (Activated values)
        :rtype: np.array
        """
        return z

    def sigmoid(self, z):
        """This is a static method of TVEKActivators used for sigmoid activator definition

        :param z: Output of the neuron before activation
        :type z: np.array

        :return: Output of neuron (Activated values)
        :rtype: np.array
        """
        return 1/(1 + np.exp(-z))