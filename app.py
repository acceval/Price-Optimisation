import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse
import json

from Model import Model 


app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('list', type=list)


model = Model('prod')

@app.route('/')
def hello():
	
	return jsonify('Welcome to Price Optimisation')



@app.route('/price_optimisation', methods=['POST'])
def price_optimisation():

	ABC = parser.parse_args()
	data_decoded = request.data.decode("utf-8") 

	#convert to json
	data_json = json.loads(data_decoded)

	# print(data_json)	

	if 'type' in data_json:
		type_ = data_json['type']
	else:
		type_ = ''

	if 'filepath' in  data_json:
		filepath = data_json['filepath']
	else:
		filepath = ''
	
	if 'features' in  data_json:
		features = data_json['features']
	else:
		features = ''

	if 'price_feature' in data_json:
		price_feature = data_json['price_feature']
	else:
		price_feature = ''

	if 'volume_feature' in  data_json:
		volume_feature = data_json['volume_feature']
	else:
		volume_feature = ''
	
	if 'product_feature' in  data_json:
		product_feature = data_json['product_feature']
	else:
		product_feature = ''

	if 'current_price' in  data_json:
		current_price = data_json['current_price']
	else:
		current_price = ''

	if 'sales_volume' in data_json:
		sales_volume = data_json['sales_volume']
	else:
		sales_volume = ''

	if 'standard_cost' in  data_json:
		standard_cost = data_json['standard_cost']
	else:
		standard_cost = ''
	
	if 'segmentation_features' in  data_json:
		segmentation_features = data_json['segmentation_features']
	else:
		segmentation_features = ''



	if type_!='' and filepath!='' and features!='' and price_feature!='' and volume_feature!='' and product_feature!='' and price_feature!='' and current_price!='' and standard_cost!='' and segmentation_features!='':

		output = model.price_optimisation(type_, filepath, features, price_feature, volume_feature, product_feature, current_price, sales_volume, standard_cost,segmentation_features)
		return jsonify(output)

	else:

		status = 0 
		error = 'There is a problem on the parameters'
		data = None

		output = dict()
		output["status"] = status
		output["error"] = error
		output["data"] = output_json

		output = json.dumps(output)

	return jsonify(output)




if __name__ == '__main__':
	port = int(os.environ.get("PORT", 5050))
	app.run(host='0.0.0.0', port = port, debug=True)

	# local
	# app.run(host='127.0.0.1', port = port, debug=True)
	