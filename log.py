import os,sys,inspect
import config, utils
from pathlib import Path

import logging 
import config 

today = utils.get_today_date()

# check if log folder exists, if not, then create it
if not os.path.exists(config.PATH_TO_FOLDER+"log"):
    os.makedirs(config.PATH_TO_FOLDER+"log")


logger = logging.getLogger(__name__)
logging.basicConfig(filename=config.PATH_TO_FOLDER+"log/loging_"+config.VERSION+"_"+today.replace('-','')+".log", level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s")
# logging.basicConfig(filename=config.PATH_TO_FOLDER+"log/loging_"+config.VERSION+".log", level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s")




class Log():
    def __init__(self,uuid=None):        

        self.uuid = uuid

    def print_(self, message):
        logging.info("{} : {} ".format(self.uuid,message))
    
