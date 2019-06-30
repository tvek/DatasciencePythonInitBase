from TVEKDNN.CustomDNN import CustomDNN
from src.logger.LOG import LOG

if "__main__" == __name__ :
    obj = CustomDNN()
    obj.generateAndInitData(50000,8)
    obj.visualizeData()

    LOG.I("Development in progress")