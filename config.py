import os

## SERVER
PATH_TO_FOLDER = str(os.getcwd())+'/'

VERSION = '0.1.0'

RANDOM_STATE = 42

# tree setting
MAX_DEPTH=3
MIN_SAMPLES_LEAF=100
CCP_ALPHA=0.001


LOCAL_URL = 'http://127.0.0.1:5050/'
HEROKU_URL = 'https://smarttradzt-price-optimisation.herokuapp.com/'

ELASTICITY_CUTOFF = -0.3
VOLUME_CONSTRAIN = 0.1


# python main.py --env local --type B2c --filepath B2C_clean.csv --features Recency Revenue_L12 Customer_Size l3y_volume standard_cost l12_sales_vol Current_Price 
# --price_feature Avg_Price_L3Y 
# --volume_feature l3y_volume 
# --product_feature Product_Group 
# --current_price Current_Price 
# --sales_volume l12_sales_vol 
# --standard_cost standard_cost 
# --segmentation_features Recency Revenue_L12 Customer_Size


var = {}

var['local'] = {}
var['local']['filepath'] = 'B2C_clean.csv'
var['local']['features'] = ['Recency','Revenue_L12','Customer_Size','l3y_volume','standard_cost','l12_sales_vol','Current_Price']
var['local']['price_feature'] = 'Avg_Price_L3Y'
var['local']['volume_feature'] = 'l3y_volume'
var['local']['product_feature'] = 'Product_Group'
var['local']['current_price'] = 'Current_Price'
var['local']['sales_volume'] = 'l12_sales_vol'
var['local']['standard_cost'] = 'standard_cost'
var['local']['segmentation_features'] = ['Recency','Revenue_L12','Customer_Size']


var['prod'] = {}
var['prod']['filepath'] = 'https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2C_clean.csv'
var['prod']['features'] = ['Recency','Revenue_L12','Customer_Size','l3y_volume','standard_cost','l12_sales_vol','Current_Price']
var['prod']['price_feature'] = 'Avg_Price_L3Y'
var['prod']['volume_feature'] = 'l3y_volume'
var['prod']['product_feature'] = 'Product_Group'
var['prod']['current_price'] = 'Current_Price'
var['prod']['sales_volume'] = 'l12_sales_vol'
var['prod']['standard_cost'] = 'standard_cost'
var['prod']['segmentation_features'] = ['Recency','Revenue_L12','Customer_Size']