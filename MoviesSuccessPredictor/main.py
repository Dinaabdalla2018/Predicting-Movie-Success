from Logic import *
from PreProcessing import *
import warnings

if __name__ == "__main__":
     warnings.filterwarnings("ignore")
     regression = True
     pre = PreProccessing(regression=regression)
     data_set = pre.moviesFile
     Logic(data_set, pre, regression=regression)

     pre = PreProccessing(regression=not regression)
     data_set = pre.moviesFile
     Logic(data_set, pre, regression=not regression)



