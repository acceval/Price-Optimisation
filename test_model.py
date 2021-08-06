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

# price_optimisation(self,type_,filepath,features,price_column,volume_column,product_column,current_price,sales_volume,standard_cost,segmentation_features):

# happy path

# def test_price_optimisation_B2C():

#     output = model.price_optimisation('B2C', filepath, features, price_feature, volume_feature, product_feature, current_price, sales_volume, standard_cost,segmentation_features)
#     output_json = json.loads(output)    

#     assert isinstance(output, str)
#     assert isinstance(output_json, dict)

#     assert output_json['status']==1
#     assert output_json['error'] is None
#     assert len(output_json['data'])>0

# def test_price_optimisation_B2B():

#     output = model.price_optimisation('B2B', filepath, features, price_feature, volume_feature, product_feature, current_price, sales_volume, standard_cost,segmentation_features)
#     output_json = json.loads(output)    

#     assert isinstance(output, str)
#     assert isinstance(output_json, dict)

#     assert output_json['status']==1
#     assert output_json['error'] is None
#     assert len(output_json['data'])>0
    


# sad path
def test_break_filepath():

    # file is not exist
    output = model.price_optimisation('B2B', 'random.csv', features, price_feature, volume_feature, product_feature, current_price, sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

    # wrong extension
    output = model.price_optimisation('B2B', 'random.file', features, price_feature, volume_feature, product_feature, current_price, sales_volume, standard_cost,segmentation_features)
    output_json = json.loads(output)    

    # print(output_json)

    assert isinstance(output, str)
    assert isinstance(output_json, dict)
    assert output_json['status']==0
    assert output_json['error'] is not None
    assert output_json['data'] is None

# sad path
def test_break_features():

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
