from TVEKDNN.ShallowDNN import ShallowDNN
from src.logger.LOG import LOG

if "__main__" == __name__ :
    obj = ShallowDNN()
    obj.generateAndInitData(50000,8)
    obj.visualizeData()
    obj.setActivationFunction("sigmoid")
    obj.model(1000, 0.1)

    LOG.I("Development in progress")