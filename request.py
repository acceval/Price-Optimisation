import requests
import json
import config, utils
import argparse
from pathlib import Path
import curlify
from Model import Model 

	
def curl_request(url,method,headers,payloads):
    # construct the curl command from request
    command = "curl -v -H {headers} {data} -X {method} {uri}"
    data = "" 
    if payloads:
        payload_list = ['"{0}":"{1}"'.format(k,v) for k,v in payloads.items()]
        data = " -d '{" + ", ".join(payload_list) + "}'"
    header_list = ['"{0}: {1}"'.format(k, v) for k, v in headers.items()]
    header = " -H ".join(header_list)
    print(command.format(method=method, headers=header, data=data, uri=url))



if __name__ == '__main__':

	# local url
	url = config.LOCAL_URL
	# url = config.HEROKU_URL

	method = 'POST'
	headers = {'Content-type': 'application/json', 'Accept': 'application/json'}

	filepath = 'https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2B_clean.csv'
	features = ["Recency", "Revenue_L12", "Customer_Size", "l3y_volume", "standard_cost", "l12_sales_vol", "Current_Price"]
	price_feature = 'Avg_Price_L3Y'
	volume_feature = 'l3y_volume'
	product_feature = 'Product_Group'
	current_price = 'Current_Price'
	sales_volume = 'l12_sales_vol'
	standard_cost = 'standard_cost'
	segmentation_features = ["Recency", "Revenue_L12", "Customer_Size"]

	function = 'price_optimisation' 
	url_ = url+function 
	data = '{"type":"B2B", "filepath" :"'+filepath+'", "features":'+str(features)+', "price_feature":"'+price_feature+'", "volume_feature":"'+volume_feature+'", "product_feature":"'+product_feature+'", "current_price":"'+current_price+'", "sales_volume":"'+sales_volume+'", "standard_cost":"'+standard_cost+'", "segmentation_features":'+str(segmentation_features)+' }'
	data = data.replace("'",'"')
	print(data)
	
	data_json = json.loads(data)
	print(url_,	data)

	send_request = requests.post(url_, data, headers=headers)

	print(curlify.to_curl(send_request.request))

	if send_request.status_code == 200:

		print(send_request.json())
	else:
		print('There is an error occurs')


	print('===================================================================')

	filepath = 'https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2C_clean.csv'

	function = 'price_optimisation' 
	url_ = url+function 
	data = '{"type":"B2C", "filepath" :"'+filepath+'", "features":'+str(features)+', "price_feature":"'+price_feature+'", "volume_feature":"'+volume_feature+'", "product_feature":"'+product_feature+'", "current_price":"'+current_price+'", "sales_volume":"'+sales_volume+'", "standard_cost":"'+standard_cost+'", "segmentation_features":'+str(segmentation_features)+' }'
	data = data.replace("'",'"')
	print(data)
	
	data_json = json.loads(data)
	print(url_,	data)

	send_request = requests.post(url_, data, headers=headers)

	print(curlify.to_curl(send_request.request))

	if send_request.status_code == 200:

		print(send_request.json())
	else:
		print('There is an error occurs')
