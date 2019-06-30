from src.TVEKDNN.CustomDNN import CustomDNN
from src.logger.LOG import LOG

def test_customdnn_objectcreation():
    """Test function for CustonDNN Object Creation

    :param None: None
    :type None: None

    :return: None
    :rtype: None
    """
    try:
        objTestableCDNN = CustomDNN()
        assert True
    except:
        assert False

def test_customdnn_objectdestruction():
    """Test function for CustonDNN Object Destruction

       :param None: None
       :type None: None

       :return: None
       :rtype: None
       """
    objTestableCDNN = CustomDNN()
    try:
        del objTestableCDNN
        assert True
    except:
        assert False
