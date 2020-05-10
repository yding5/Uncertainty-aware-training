
import os

class PytorchExperimentLogger(object):
    """
    A single class for logging your pytorch experiments to file.
    Extends the ExperimentLogger also also creates a experiment folder with a file structure:
    """

    def __init__(self, saveDir):

        self.saveFile = os.path.join(saveDir, "elog.txt")

    def print(self, strT):
        print(strT)
        f = open(self.saveFile, 'a')
        f.writelines(strT+'\n')
        f.close()



