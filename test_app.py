import os
import pytest
import json
import requests
import config
from config import var
from Model import Model 
from app import app

env = 'local'

# default vars
filepath = 'B2B_clean.csv'
features = ["Recency", "Revenue_L12", "Customer_Size", "l3y_volume", "standard_cost", "l12_sales_vol", "Current_Price"]
price_feature = 'Avg_Price_L3Y'
volume_feature = 'l3y_volume'
product_feature = 'Product_Group'
current_price = 'Current_Price'
sales_volume = 'l12_sales_vol'
standard_cost = 'standard_cost'
segmentation_features = ["Recency", "Revenue_L12", "Customer_Size"]

model = Model(env)
# local url
url = config.LOCAL_URL
# url = config.HEROKU_URL


def test_price_optimisation_B2B_app(app, client):
            
    function = 'price_optimisation' 
    url_ = url+function 
    data = '{"type":"B2B", "filepath" :"'+filepath+'", "features":'+str(features)+', "price_feature":"'+price_feature+'", "volume_feature":"'+volume_feature+'", "product_feature":"'+product_feature+'", "current_price":"'+current_price+'", "sales_volume":"'+sales_volume+'", "standard_cost":"'+standard_cost+'", "segmentation_features":'+str(segmentation_features)+' }'
    data = data.replace("'",'"')
    
    send_request = client.post(url_, data=data, follow_redirects=True)    

    assert send_request.status_code == 200


def test_price_optimisation_B2C_app(app, client):
    
    filepath = 'B2C_clean.csv'

    function = 'price_optimisation' 
    url_ = url+function 
    data = '{"type":"B2B", "filepath" :"'+filepath+'", "features":'+str(features)+', "price_feature":"'+price_feature+'", "volume_feature":"'+volume_feature+'", "product_feature":"'+product_feature+'", "current_price":"'+current_price+'", "sales_volume":"'+sales_volume+'", "standard_cost":"'+standard_cost+'", "segmentation_features":'+str(segmentation_features)+' }'
    data = data.replace("'",'"')
    
    send_request = client.post(url_, data=data, follow_redirects=True)    

    assert send_request.status_code == 200


