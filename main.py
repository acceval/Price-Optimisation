import os,sys,inspect,getopt,io
from pathlib import Path
import argparse

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from log import Log
import config, utils

import pandas as pd
import numpy as np
import json
import string

from Model import Model 


if __name__== '__main__':

	start = utils.get_time()
	print(start)
	
	today = None

	parser = argparse.ArgumentParser()	
	parser.add_argument("--env", "-e", help="State the environment", required=True)	
	parser.add_argument("--type", "-t", help="State the type", required=True)	
	parser.add_argument("--filepath", "-f", help="Path to the input file", required=True)	
	parser.add_argument("--features", "-ff", nargs="*", help="Features that will be used in the calculation, should not contain space", required=True)	
	parser.add_argument("--segmentation_features", "-sf", nargs="*", help="Segmentation Features that will be used in the calculation, should not contain space", required=True)	
	parser.add_argument("--price_feature", "-pri", help="Price feature, should not contain space", required=True)	
	parser.add_argument("--volume_feature", "-v", help="Volume feature, should not contain space", required=True)	
	parser.add_argument("--product_feature", "-pro", help="Product feature, should not contain space", required=True)	
	parser.add_argument("--current_price", "-c", help="Current Price, should not contain space", required=True)	
	parser.add_argument("--sales_volume", "-sv", help="Sales Volume, should not contain space", required=True)	
	parser.add_argument("--standard_cost", "-sc", help="Standard Cost, should not contain space", required=True)	

	args = parser.parse_args()

	env = None
	if args.env is None:
		print("State the environment!!")
	else:
		env = args.env
	
	type_ = None
	if args.type is None:
		print("State the type!!")
	else:
		type_ = args.type

	filepath = None
	if args.filepath is None:
		print("State the filepath!!")
	else:
		filepath = args.filepath

	features = None
	if args.features is None:
		print("State the features!!")
	else:
		features = args.features	

	segmentation_features = None
	if args.segmentation_features is None:
		print("State the segmentation_features!!")
	else:
		segmentation_features = args.segmentation_features

	price_feature = None
	if args.features is None:
		print("State the price feature!!")
	else:
		price_feature = args.price_feature

	volume_feature = None
	if args.volume_feature is None:
		print("State the volume feature!!")
	else:
		volume_feature = args.volume_feature

	product_feature = None
	if args.product_feature is None:
		print("State the product feature!!")
	else:
		product_feature = args.product_feature

	current_price = None
	if args.current_price is None:
		print("State the current price!!")
	else:
		current_price = args.current_price

	sales_volume = None
	if args.sales_volume is None:
		print("State the sales volume!!")
	else:
		sales_volume = args.sales_volume

	standard_cost = None
	if args.standard_cost is None:
		print("State the standard cost!!")
	else:
		standard_cost = args.standard_cost


	print('env:',env)
	print('filepath:',filepath)
	print('features:',features)	
	print('price_feature:',price_feature)
	print('volume_feature:',volume_feature)
	print('product_feature:',product_feature)
	print('current_price:',current_price)
	print('sales_volume:',sales_volume)
	print('standard_cost:',standard_cost)


	print('-------------------------------------------')
	
	log = Log()		

	msg = __name__+'.'+utils.get_function_caller()
	log.print_(msg)

	
	model = Model(env)

	# price_optimisation(self, type_, filepath, features, price_column, volume_column, product_column, current_price_column, sales_volume, standard_cost):
	output = model.price_optimisation('B2C', filepath, features, price_feature, volume_feature, product_feature, current_price, sales_volume, standard_cost,segmentation_features)
	print(type(output))
	print(output)

	
	print('\n\n\n')
	print('==============================================================================================')

	output = model.price_optimisation('B2B','B2B_clean.csv',features,price_feature,volume_feature,product_feature,current_price,sales_volume,standard_cost,segmentation_features)
	print(type(output))
	print(output)
	





	print('-------------------------------------------')

	end = utils.get_time()
	print(end)

	print(end - start)


	msg = 'start:',start
	log.print_(msg)

	msg = 'end:',end
	log.print_(msg)

	msg = 'total:',end-start
	log.print_(msg)	
	