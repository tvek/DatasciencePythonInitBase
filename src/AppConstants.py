import sys
import os

class AppConstants(object):

    @staticmethod
    def getRootPath():
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'

