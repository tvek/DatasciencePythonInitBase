import logging
import sys
import os

class LOG():
    '''
    A class used for centrallized custom logging
    '''

    stLogger = logging.getLogger(__name__)
    """loggering.Logger: Logger variable used internally

    This is the type of logger which is being applied accross the system
    """

    def __init__(self):
        """
        __init__() is a instance method of LOG used initialize the class instance
        """

        self.Configure()


    @staticmethod
    def ConfigureHandler(handler):
        """This function will apply the generic configuration to the input handler

        :param handler: Logger Handler Instance
        :type handler: logger.Logger.Handler

        :return: Instance of Logger after applying the configuration
        :rtype: logger.Logger.Handler
        """
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)7s - %(message)s',"%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        return handler

    @staticmethod
    def Configure():
        """This function is used to configure the logger

        :return: No Return
        :rtype: None
        """
        stLogger = logging.getLogger(__name__)
        stLogger.setLevel(logging.INFO)

        # create a file handler
        handler = LOG.ConfigureHandler(logging.FileHandler(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'../../logs/CustomDNN.log'))
        stLogger.addHandler(handler)

        # create a stdout handler
        handler = LOG.ConfigureHandler(logging.StreamHandler(sys.stdout))
        stLogger.addHandler(handler)
        LOG.stLogger.info("Logger Started Successfully")

    @staticmethod
    def E(string):
        """This function will log to error channel

        :param string: String to be printed to error channel
        :type string: string

        :return: None
        :rtype: None
        """
        LOG.stLogger.error(string)

    @staticmethod
    def D(string):
        """This function will log to debug channel

        :param string: String to be printed to debug channel
        :type string: string

        :return: None
        :rtype: None
        """
        LOG.stLogger.debug(string)

    @staticmethod
    def I(string):
        """This function will log to information channel

        :param string: String to be printed to information channel
        :type string: string

        :return: None
        :rtype: None
        """
        LOG.stLogger.info(string)

    @staticmethod
    def W(string):
        """This function will log to information channel

        :param string: String to be printed to information channel
        :type string: string

        :return: None
        :rtype: None
        """
        LOG.stLogger.warn(string)