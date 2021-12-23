import os

## SERVER
# PATH_TO_FOLDER = '/home/cst/Jixie/source code/unit_forecasting/'
PATH_TO_FOLDER = str(os.getcwd())+'/'


VERSION = '0.1.0'

RANDOM_STATE = 42

SL = 0.05

# tree setting
MAX_DEPTH=3
MIN_SAMPLES_LEAF=100
CCP_ALPHA=0.001


LOCAL_URL = 'http://127.0.0.1:5050/'
HEROKU_URL = 'https://price-segmentation.herokuapp.com/'



var = {}

var['local'] = {}
var['local']['filepath'] = 'sample_input_file.csv'
var['local']['features'] = ['Customer_Type', 'Customer_Industry', 'Grade', 'Country', 'Destination_Port', 'City_State', 'Shipping_Condition', 'Export/Domestic', 'QUANTITY']
var['local']['target_feature'] = 'Price_Premium'
var['local']['index'] = 'Index'
var['local']['price_per_segment'] = 'price_per_segment.json'
var['local']['global_threshold'] = 'global_threshold.json'
var['local']['customised_threshold'] = 'customised_threshold_final.json'
var['local']['price_power_index_threshold']	 = 'price_power_index_final.json'
var['local']['bad_price_per_segment'] = 'https://raw.githubusercontent.com/acceval/Price-Segmentation/main/bad_price_per_segment.json'


var['prod'] = {}
var['prod']['filepath'] = 'https://raw.githubusercontent.com/acceval/Price-Segmentation/main/sample_input_file.csv'
var['prod']['features'] = ['Customer_Type', 'Customer_Industry', 'Grade', 'Country', 'Destination_Port', 'City_State', 'Shipping_Condition', 'Export/Domestic', 'QUANTITY']
var['prod']['target_feature'] = 'Price_Premium'
var['prod']['index'] = 'Index'
var['prod']['price_per_segment'] = 'https://raw.githubusercontent.com/acceval/Price-Segmentation/main/price_per_segment.json'
var['prod']['global_threshold'] = 'https://raw.githubusercontent.com/acceval/Price-Segmentation/main/global_threshold.json'
var['prod']['customised_threshold'] = 'https://raw.githubusercontent.com/acceval/Price-Segmentation/main/customised_threshold_final.json'
var['prod']['price_power_index_threshold']	 = 'https://raw.githubusercontent.com/acceval/Price-Segmentation/main/price_power_index_final.json'
var['prod']['bad_price_per_segment'] = 'https://raw.githubusercontent.com/acceval/Price-Segmentation/main/bad_price_per_segment.json'
