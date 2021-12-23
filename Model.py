import os,sys,inspect,getopt,io 
from pathlib import Path
import argparse

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from log import Log
import config, utils

import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import json
import string
import importlib
import importlib.util
from import_file import import_file
import requests

from sklearn.feature_selection import f_regression
from sklearn.tree import DecisionTreeRegressor, _tree, export_text
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

class Model:

	def __init__(self, env='local'):
		
		self.uuid = utils.get_uuid()
		self.log = Log(self.uuid)		

		self.env = env

		# tree
		self.max_depth = config.MAX_DEPTH
		self.min_samples_leaf = config.MIN_SAMPLES_LEAF
		self.ccp_alpha = config.CCP_ALPHA
		self.random_state = config.RANDOM_STATE


	def status_check(self,filepath:string,features:list, target, index=None):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		status = 0
		result = None
		error = None

		# check if the file is csv file
		try:

			ext = filepath.split('.')[-1]			

			if ext.lower()=='csv':

				msg = 'File is a csv file'
				self.log.print_(msg)
				print(msg)

				status = 1

			else:

				msg = 'File is not a csv file'
				self.log.print_(msg)
				print(msg)

				error = msg

				return status, error

		except Exception as e:

			msg = 'Error when checking the file extension'
			self.log.print_(msg)
			print(msg)


		if status:

			# check if the all features in the file
			try:

				try:

					data = pd.read_csv(filepath)
					
				except Exception as e:

					msg = 'Cannot read the input file'
					self.log.print_(msg)
					print(msg)

					error = msg

					return status, error

				all_features = data.columns

				# # dataframe size should be >1
				if data.shape[0]:

					msg = 'Dataframe shape:',data.shape
					self.log.print_(msg)
					print(msg)

				else:

					msg = 'DataFrame has no data'
					self.log.print_(msg)
					print(msg)

					status = 0
					error = msg

					return status, error


				# check if index exists and unique
				if index is not None:

					if index in all_features:

						msg = 'Index exists'
						self.log.print_(msg)
						print(msg)

					else: 
						
						msg = 'Index does not exist'
						self.log.print_(msg)
						print(msg)

						status = 0
						error = msg

						return status, error

					if not data[index].is_unique:

						msg = 'Index does not have unique values'
						self.log.print_(msg)
						print(msg)

						status = 0
						error = msg

						return status, error
					

				# check if target exists and type is float
				# print(target)
				# print(all_features)
				if target in all_features:

					msg = 'Target feature exists'
					self.log.print_(msg)
					print(msg)

				else:

					msg = 'Target feature does not exist'
					self.log.print_(msg)
					print(msg)

					status = 0
					error = msg

					return status, error
				
				if 'float' in str(data[target].dtype):

					msg = 'Target feature type is acceptable'
					self.log.print_(msg)
					print(msg)

				else:

					msg = 'Target feature type is not acceptable'
					self.log.print_(msg)
					print(msg)

					status = 0
					error = msg

					return status, error
					
				# check if features sub of all_features
				if not set(features).issubset(set(all_features)):

					msg = 'All or some of the features do not exists'
					self.log.print_(msg)
					print(msg)

					status = 0
					error = msg

					return status, error
				
				status = 1

				return status, error
				

			except Exception as e:

				msg = 'Error when checking the features, target feature and index'
				self.log.print_(msg)
				print(msg)

				status = 0

		return status		


	def feature_engineering(self,dataframe,features,target):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		categorical_features = []
		numerical_features = []

		for col,type_ in zip(dataframe[features].columns,dataframe[features].dtypes):
			
			if str(type_)=='object' and col!=target:        
				categorical_features.append(col)        
			else:
				numerical_features.append(col)

		# handle empty values
		for categorical_feature in categorical_features:

			dataframe[categorical_feature].fillna('No data', inplace=True)

		for numerical_feature in numerical_features:

			dataframe[numerical_feature].fillna(dataframe[numerical_feature].median(), inplace=True)


		# feature engineering
		for categorical_feature in categorical_features:

			dataframe[categorical_feature] = dataframe.groupby(categorical_feature)[target].transform('max')

		return dataframe
			

	def backwardElimination(self, x, Y, sl, columns):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		numVars = len(x[0])
		pvals = []
		for i in range(0, numVars):
			regressor_OLS = sm.OLS(Y, x).fit()
			maxVar = max(regressor_OLS.pvalues).astype(float)
			if maxVar > sl:
				for j in range(0, x.shape[1] - i):
					if (regressor_OLS.pvalues[j].astype(float) == maxVar):
						x = np.delete(x, j, 1)
						columns = np.delete(columns, j)
			else:
				pvals.append(maxVar)
					   
		regressor_OLS.summary()
		return x, columns, pvals


	def features_assessment(self,filepath:string,features:list,target:string,SL=0.05):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		params = locals()
		msg = 'params:'+str(params)
		self.log.print_(msg)

		# print('target:',target)

		return_ = dict()		

		if isinstance(filepath, str) and isinstance(features, list) and isinstance(target, str):

			target = target.strip()

			status, error = self.status_check(filepath,features,target)

			if status :

				try:

					data = pd.read_csv(filepath)

					# status, target = self.check_target(target)

					if status:

						# handle outliers
						data = data[
									(
										(data[target]>=(data[target].mean() - (3*data[target].std())))
										& (data[target]<=(data[target].mean() + (3*data[target].std())))
									)
									]

						# do the feature assement

						data = self.feature_engineering(data,features,target)
						
						selected_columns = features
						data_modeled, selected_columns, pvals = self.backwardElimination(data[features].values, data[target].values, SL, selected_columns)

						# print(selected_columns, pvals)

						feature_pval = {}
						if len(selected_columns)==len(pvals):
							for selected_column, pval in zip(selected_columns,pvals):
								feature_pval[selected_column] = str(round(pval, 3))
						
						for feature in features:
							if feature not in feature_pval.keys():
								feature_pval[feature] = '>'+str(SL)

						# if len(features)==1:

						# 	freg = f_regression(np.array(data[features].values.reshape(-1, 1),dtype=float), np.array(data[target].values.ravel(),dtype=float))

						# else:

						# 	freg = f_regression(np.array(data[features].values,dtype=float), np.array(data[target].values.ravel(),dtype=float))

						# p_values = freg[1]

						# feature_pval = dict()

						# for p_value, feature in zip(p_values,features):
				
						# 	feature_pval[feature] = round(p_value, 3)    

						
						result = json.dumps(feature_pval)

					else:

						msg = 'Error on target feature'
						self.log.print_(msg)
						print(msg)

						status = 0	
						result = None



				except:

					msg = 'Error when doing features_assessment'
					self.log.print_(msg)
					print(msg)

					status = 0	
					result = None



		else:

			msg = 'Parameters type are wrong'
			self.log.print_(msg)
			print(msg)

			status = 0	
			error = msg
			result = None


		return_["status"] = status
		return_["error"] = error

		if status==1:

			return_["data"] = result

		else:

			return_["data"] = None			

		return_json = json.dumps(return_)

		return return_json

	def export_py_code(self, tree, feature_names, max_depth=100, spacing=4):

		if spacing < 2:
			raise ValueError('spacing must be > 1')

		# Clean up feature names (for correctness)
		nums = string.digits
		alnums = string.ascii_letters + nums
		clean = lambda s: ''.join(c if c in alnums else '_' for c in s)
		features = [clean(x) for x in feature_names]
		features = ['_'+x if x[0] in nums else x for x in features if x]
		if len(set(features)) != len(feature_names):
			raise ValueError('invalid feature names')

		# First: export tree to text
		res = export_text(tree, feature_names=features, 
						max_depth=max_depth,
						decimals=6,
						spacing=spacing-1)

		# Second: generate Python code from the text
		skip, dash = ' '*spacing, '-'*(spacing-1)
		code = 'def decision_tree({}):\n'.format(', '.join(features))
		for line in repr(tree).split('\n'):
			code += skip + "# " + line + '\n'
		for line in res.split('\n'):
			line = line.rstrip().replace('|',' ')
			if '<' in line or '>' in line:
				line, val = line.rsplit(maxsplit=1)
				line = line.replace(' ' + dash, 'if')
				line = '{} {:g}:'.format(line, float(val))
			else:
				line = line.replace(' {} class:'.format(dash), 'return')
			code += skip + line + '\n'

		return code.replace('- value:','return').replace('-','')

	def get_func_args(self,f):

		if hasattr(f, 'args'):
			return f.args
		else:
			return list(inspect.signature(f).parameters)


	def segmentation(self,filepath:string,features:list,target:string, index:string, max_depth=None, min_samples_leaf=None):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		params = locals()
		msg = 'params:'+str(params)
		self.log.print_(msg)
		
		# tree
		if max_depth is not None:
			self.max_depth = max_depth

		if min_samples_leaf is not None:
			self.min_samples_leaf = min_samples_leaf


		return_ = dict()

		if isinstance(filepath, str) and isinstance(features, list) and isinstance(target, str) or isinstance(filepath, str) and isinstance(features, list) and isinstance(target, str) and isinstance(index, str):

			target = target.strip()

			if index is not None:
				index = index.strip()

			status, error = self.status_check(filepath,features,target,index)		

			if status:

				try:				

					data = pd.read_csv(filepath)
					
					# handle outliers		
					data = data[
								(
									(data[target]>=(data[target].mean() - (3*data[target].std())))
									& (data[target]<=(data[target].mean() + (3*data[target].std())))
								)
								]

					data = self.feature_engineering(data,features,target)

					clf = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=self.random_state, ccp_alpha=self.ccp_alpha)
					clf.fit(data[features],data[target])		

					# get the tree rules
					res = self.export_py_code(clf, feature_names=features, max_depth=self.max_depth, spacing=4)					

					# save the function as a file
					filename = '_'+utils.get_unique_filename('tree')+'.py'
					
					with open(filename, 'w') as f:
						f.write(res)

					module = import_file(filename)					
					
					# rename column names
					for a,b in zip(features,self.get_func_args(module.decision_tree)): 
						data.rename(columns={a:b},inplace=True)


					data['node'] = data[self.get_func_args(module.decision_tree)].apply(lambda x: module.decision_tree(*x), axis=1)
					data['node'] = data['node'].apply(lambda x:str(x).replace('[','').replace(']',''))
					
					# remove the file
					os.remove(filename)

					# encode the segment
					le = LabelEncoder()
					data['segment'] = le.fit_transform(data['node'])


					output_df = data[[index]+['segment']]
					output_df = output_df.groupby(['segment'])[index].apply(list).reset_index() 
					

					# convert dataframe to json
					# output_json = output_df.to_json(orient ='records')
					outputs = []

					for i, rows in enumerate(output_df.itertuples(),1):

						_str = '{"segment":'+str(rows[1])+', "'+index+'":'+str(rows[2])+'}'
						outputs.append(_str)

					final_output = '['+', '.join(outputs)+']'
					final_output = final_output.replace("'",'"')					
					output_json = json.loads(final_output)



				except Exception as e:

					msg = 'Error on modeling'
					self.log.print_(msg)
					print(msg)

					print(e)

				
			else:

				msg = 'There is problem on the file path, features, target feature, or index'
				self.log.print_(msg)
				print(msg)

		else:

			msg = 'Parameters type are wrong'
			self.log.print_(msg)
			print(msg)

			status = 0	
			error = msg
			result = None


		
		return_["status"] = status
		return_["error"] = error

		if status==1:

			return_["data"] = output_json

		else:

			return_["data"] = None			

		return_ = json.dumps(return_)

		# # save this json for testing purpose
		# with open('sample_json.json', 'w') as f:
		# 	f.write(return_)

		return return_


	def get_threshold(self,price_threshold,is_power_index,segment_field,segment=None):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		try :

			if (is_power_index==True and isinstance(price_threshold, list)):

				threshold = [threshold for threshold in price_threshold if threshold[segment_field]==segment]

				if len(threshold)==1:
					
					threshold = threshold[0]['threshold'][0]
					return threshold

				else:
					raise Exception('There is an error in the JSON threshold file')



			elif (is_power_index==False and isinstance(price_threshold, dict)):

				return price_threshold



		except Exception as e:

			msg = 'There is an error in the JSON threshold file'
			self.log.print_(msg)
			print(msg)

			status = 0	
			error = msg
			result = None							

	def read_json(self,json_file):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		if isinstance(json_file, str):

			try:

				if self.env == 'local':

					with open(json_file) as f:
						json_output = json.load(f)

				elif self.env == 'prod':

					resp = requests.get(json_file)
					json_output = json.loads(resp.text)   

			except Exception as e:

				msg = 'There is an error when reading JSON file'
				self.log.print_(msg)
				print(msg)

			finally:
				
				return json_output

	def get_threshold(self, threshold_item, key):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		try:

			if isinstance(threshold_item, dict) and isinstance(key, str):

				return threshold_item[key]

		except Exception as e:

			msg = 'There is problem with the threshold item'
			self.log.print_(msg)
			print(msg)

	def get_price_per_segment(self,price_per_segment, segment, target):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		try:

			if isinstance(price_per_segment, list) and (isinstance(target,str)):

				return price_per_segment[segment][target]

		except Exception as e:

			msg = 'There is problem with the threshold item'
			self.log.print_(msg)
			print(msg)

	def segment_check(self,price_per_segment,threshold,segment):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		A = [item[segment] for item in price_per_segment]					
		B = [list(utils.find(segment, item))[0] for item in threshold ]

		if len(A)==len(B) and set(A) == set(B):

			return True

		else:

			return False



	def price_segmentation(self, price_per_segment, threshold, segment, target):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		params = locals()
		msg = 'params:'+str(params)
		self.log.print_(msg)


		return_ = dict()

		status = 0
		error = None

		if isinstance(price_per_segment, str) and isinstance(threshold, str) and isinstance(segment, str) and isinstance(target, str):

			price_per_segment = price_per_segment.strip()
			threshold = threshold.strip()
			segment = segment.strip()
			target = target.strip()

			# check if number of segments in price per segment = number of segments in threshold

			# read json 				
			try:

				price_per_segment_json = self.read_json(price_per_segment)
				threshold_json = self.read_json(threshold)							

			except Exception as e:

				msg = 'Error on JSON file'
				self.log.print_(msg)
				print(msg)

				print(e)

				status = 0 
				error = msg

			else:

				# print(price_per_segment_json)				
				price_per_segment_df = pd.DataFrame(price_per_segment_json)
				# print(price_per_segment_df)

				if (segment in price_per_segment_df.columns):
				
					if target in price_per_segment_df.columns:
				
						# check json level of depth
						depth = utils.depth(threshold_json)

						segments = price_per_segment_df[segment].unique()

						try:

							if not segment in threshold_json and depth==1:

								output = list()
								for i,segment_ in enumerate(segments):

									line = list()    
									line.append('"'+segment+'":'+str(segment_))
																
									prices = self.get_price_per_segment(price_per_segment_json, segment_,target)

									_str = []
									for k,v in threshold_json.items():
										
										tmp = '{"'+str(k)+'":'+str(np.percentile(prices, v*10))+'}'        
										_str.append(tmp)
																		
									# _str = '"'+target+'":['+','.join(_str)+']'								
									_str = '"threshold":['+','.join(_str)+']'								

									line.append(_str)
									line = ','.join(line)								
									line = "{"+line+"}"
									
									output.append(line)
								
								output = '['+','.join(output)+']'

								result = json.loads(output)

								# print(result)


								status = 1


							else:


								if self.segment_check(price_per_segment_json,threshold_json,segment) :


									# print('depth:',depth)
									# print(threshold_json)
									threshold_json_str = str(threshold_json)

									# print('Original:',threshold_json_str)

									# result = []

									for i, segment_ in enumerate(segments):

										prices = self.get_price_per_segment(price_per_segment_json, segment_,target)

										item = threshold_json[segment_]
										item_str = str(item)
										new_item_str = str(item)

										# print(type(item))
										# print('Original item:',item)
										# print('item_str:',item_str)
										
										thresholds = list(utils.find('threshold', item))
										# print(len(thresholds),thresholds)
										
										# print('\n\n\n')

										for threshold in thresholds:

											threshold_ = threshold[0]		
											original_threshold_str = str(threshold_)

											# print('original_threshold_str:',original_threshold_str)							

											for k,v in threshold_.items():

												threshold_.update({k: np.percentile(prices, v*10)})

											# print('new values: ',str(threshold_))
											# print('Replace: ',original_threshold_str,'with',str(threshold_),'on',item_str)

											# update the original item using string operation
											new_item_str = new_item_str.replace(original_threshold_str,str(threshold_))

											# print('Before replacement:',item_str)
											# print('After replacement:',new_item_str)



											# result.append(item_str)
											# print('\n\n\n')


										# print('replace',item_str,'with',new_item_str)
										threshold_json_str = threshold_json_str.replace(item_str,new_item_str) 



										# print('***************************************************')
									# print('final:',threshold_json_str)

									# # result = json.loads(result)
									# result = '['+','.join(threshold_json_str).replace("'",'"')+']'
									result = threshold_json_str.replace("'",'"')
									result = json.loads(result)
									# print(result)
									status = 1

								else:

									msg = 'Segment are not same'
									self.log.print_(msg)
									print(msg)

									status = 0	
									error = msg
									result = None
						except Exception as e:

							msg = 'Error when do price segmentation'
							self.log.print_(msg)
							print(msg)

							print(e)

							status = 0	
							error = msg
							result = None




					else:

							msg = 'Target does not exist'
							self.log.print_(msg)
							print(msg)

							status = 0	
							error = msg
							result = None

				else:

						msg = 'Segment does not exist'
						self.log.print_(msg)
						print(msg)

						status = 0	
						error = msg
						result = None

		else:

			msg = 'Parameters type are wrong'
			self.log.print_(msg)
			print(msg)

			status = 0	
			error = msg
			result = None


		return_["status"] = status
		return_["error"] = error

		if status==1:

			return_["data"] = result

		else:

			return_["data"] = None			

		return_ = json.dumps(return_)

		return return_







				
	# def price_segmentation(self, price_per_segment, price_threshold, segment, target, is_power_index=False):

	# 	msg = self.__class__.__name__+'.'+utils.get_function_caller()
	# 	self.log.print_(msg)
	# 	print(msg)

	# 	return_ = dict()

	# 	status = 0
	# 	error = None

	# 	if isinstance(price_per_segment, str) and isinstance(price_threshold, str) and isinstance(is_power_index, bool):
			
	# 		try:

	# 			price_per_segment = price_per_segment.strip()
	# 			price_threshold = price_threshold.strip()
				

	# 			# read json 				
	# 			try:

	# 				price_per_segment_json = self.read_json(price_per_segment)
	# 				price_threshold_json = self.read_json(price_threshold)
				

	# 			except Exception as e:

	# 				msg = 'Error on JSON file'
	# 				self.log.print_(msg)
	# 				print(msg)

	# 				print(e)

	# 				status = 0 
	# 				error = msg

	# 			else:


	# 				if (is_power_index==False and isinstance(price_threshold_json, dict)) or (is_power_index==True and isinstance(price_threshold_json, list)):

	# 					price_per_segment_df = pd.DataFrame(price_per_segment_json)

	# 					# check if segment and target exists
	# 					if segment in price_per_segment_df.columns:

	# 						if target in price_per_segment_df.columns:


	# 							output = list()
	# 							for i,segment_ in enumerate(price_per_segment_df[segment].unique()):

	# 								line = list()    
	# 								line.append('"'+segment+'":'+str(segment_))
									  
	# 								sub = price_per_segment_df[price_per_segment_df[segment]==segment_]

	# 								prices = sub[target].values[0]
													

	# 								if is_power_index:
	# 									threshold = self.get_threshold(price_threshold_json,is_power_index,segment_field=segment, segment=segment_)
	# 								else:
	# 									threshold = self.get_threshold(price_threshold_json,is_power_index,segment_field=segment)



	# 								_str = []
	# 								for k,v in threshold.items():
										
	# 									tmp = '{"'+str(k)+'":'+str(np.percentile(prices, v*10))+'}'        
	# 									_str.append(tmp)
										
									
	# 								_str = '"'+target+'":['+','.join(_str)+']'
									

	# 								line.append(_str)
	# 								line = ','.join(line)
	# 								line = "{"+line+"}"
									
									
	# 								output.append(line)					    
									
	# 							output = '['+','.join(output)+']'

	# 							result = json.loads(output)

	# 							status = 1



	# 						else:

	# 							msg = 'Target does not exist'
	# 							self.log.print_(msg)
	# 							print(msg)

	# 							status = 0	
	# 							error = msg
	# 							result = None
	# 					else:

	# 						msg = 'There is an error in the JSON threshold file'
	# 						self.log.print_(msg)
	# 						print(msg)

	# 						status = 0	
	# 						error = msg
	# 						result = None

	# 				else:

	# 					msg = 'Segment does not exist'
	# 					self.log.print_(msg)
	# 					print(msg)

	# 					status = 0	
	# 					error = msg
	# 					result = None

					
	# 		except Exception as e:

	# 			msg = 'Error when do price segmentation!!'
	# 			self.log.print_(msg)
	# 			print(msg)

	# 			print(e)


			




	# 	else:

	# 		msg = 'Parameters type are wrong'
	# 		self.log.print_(msg)
	# 		print(msg)

	# 		status = 0	
	# 		error = msg
	# 		result = None


	# 	return_["status"] = status
	# 	return_["error"] = error

	# 	if status==1:

	# 		return_["data"] = result

	# 	else:

	# 		return_["data"] = None			

	# 	return_ = json.dumps(return_)

	# 	return return_
