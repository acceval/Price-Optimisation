import os
import pytest
from Model import Model 
import json
import requests
from config import var

env = 'prod'

# default vars
filepath = var[env]['filepath']
features = var[env]['features']
price_feature = var[env]['price_feature']
volume_feature = var[env]['volume_feature']
product_feature = var[env]['product_feature']
current_price = var[env]['current_price']
sales_volume = var[env]['sales_volume']
standard_cost = var[env]['standard_cost']
segmentation_features = var[env]['segmentation_features']


model = Model(env)

# happy path

def test_price_optimisation_B2C():


    output = model.price_optimisation('B2C', filepath, features, price_feature, volume_feature, product_feature, current_price, sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)

    assert output_json['status']==1
    assert output_json['error'] is None
    assert len(output_json['data'])>0

def test_price_optimisation_B2B():

    filepath = 'https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2B_clean.csv'    

    output = model.price_optimisation('B2B', filepath, features, price_feature, volume_feature, product_feature, current_price, sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)

    assert output_json['status']==1
    assert output_json['error'] is None
    assert len(output_json['data'])>0
    

# sad path
def test_break_filepath():
    
    filepath = 'https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2B_clean.csv'    

    # file is not exist
    output = model.price_optimisation('B2B', 'random.csv', features, price_feature, volume_feature, product_feature, current_price, sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

    # wrong extension
    output = model.price_optimisation('B2B','random.file', features, price_feature, volume_feature, product_feature, current_price, sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    # print(output_json)

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

# sad path
def test_break_features():

    filepath = 'https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2B_clean.csv'

    # features is a string
    output = model.price_optimisation('B2B', filepath, 'features', price_feature, volume_feature, product_feature, current_price, sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

    # random features
    output = model.price_optimisation('B2B', filepath, ['features'], price_feature, volume_feature, product_feature, current_price, sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

def test_break_price():

    filepath = 'https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2B_clean.csv'

    # price_feature is a list
    output = model.price_optimisation('B2B', filepath, features, ['price_feature'], volume_feature, product_feature, current_price, sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

    # random price_feature
    output = model.price_optimisation('B2B', filepath, features, 'random_price_feature', volume_feature, product_feature, current_price, sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

def test_break_volume():

    filepath = 'https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2B_clean.csv'

    # price_feature is a list
    output = model.price_optimisation('B2B', filepath, features, price_feature, ['volume_feature'], product_feature, current_price, sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

    # random price_feature
    output = model.price_optimisation('B2B', filepath, features, price_feature, 'random_volume_feature', product_feature, current_price, sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

def test_break_product():

    filepath = 'https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2B_clean.csv'

    # product is a list
    output = model.price_optimisation('B2B', filepath, features, price_feature, volume_feature, ['product_feature'], current_price, sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

    # random product
    output = model.price_optimisation('B2B', filepath, features, price_feature, volume_feature, 'random_product_feature', current_price, sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None


def test_break_current_price():

    filepath = 'https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2B_clean.csv'

    # current_price is a list
    output = model.price_optimisation('B2B', filepath, features, price_feature, volume_feature, product_feature, ['current_price'], sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

    # random current_price
    output = model.price_optimisation('B2B', filepath, features, price_feature, volume_feature, product_feature, 'random_current_price', sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None


def test_break_sales_volume():

    filepath = 'https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2B_clean.csv'

    # sales_volume is a list
    output = model.price_optimisation('B2B', filepath, features, price_feature, volume_feature, product_feature, current_price, ['sales_volume'], standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

    # random current_price
    output = model.price_optimisation('B2B', filepath, features, price_feature, volume_feature, product_feature, current_price, 'random_sales_volume', standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None


def test_break_standard_cost():

    filepath = 'https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2B_clean.csv'

    # sales_volume is a list
    output = model.price_optimisation('B2B', filepath, features, price_feature, volume_feature, product_feature, current_price, sales_volume, ['standard_cost'], segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

    # random current_price
    output = model.price_optimisation('B2B', filepath, features, price_feature, volume_feature, product_feature, current_price, sales_volume, 'random_standard_cost',segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

def test_break_segmentation_features():

    filepath = 'https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2B_clean.csv'

    # segmentation_features is a string
    output = model.price_optimisation('B2B', filepath, features, price_feature, volume_feature, product_feature, current_price, sales_volume, standard_cost, 'segmentation_features')
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

    # random segmentation_features
    output = model.price_optimisation('B2B', filepath, features, price_feature, volume_feature, product_feature, current_price, sales_volume, standard_cost,['a','b'])
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None
