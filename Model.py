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
import importlib
import importlib.util
from import_file import import_file
import requests
from scipy.optimize import minimize

from sklearn.tree import DecisionTreeRegressor, _tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

class Model:

	def __init__(self, env='local'):
		

		self.log = Log()		

		self.env = env
		
		# tree
		self.max_depth = config.MAX_DEPTH
		self.min_samples_leaf = config.MIN_SAMPLES_LEAF
		self.ccp_alpha = config.CCP_ALPHA
		self.random_state = config.RANDOM_STATE

		self.elasticity_cutoff = config.ELASTICITY_CUTOFF
		self.volume_constrain = config.VOLUME_CONSTRAIN

	def group_features(self,dataframe,features):

		categorical_features = []
		numerical_features = []


		for col,type_ in zip(dataframe[features].columns,dataframe[features].dtypes):

			if str(type_)=='object':        
				categorical_features.append(col)        
			else:
				numerical_features.append(col)

		return categorical_features, numerical_features

	def target_encoding(self,data,categorical_features,price_column):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		for col in categorical_features:

			data[col] = data.groupby(col)[target].transform('median')

		return data


	def segmentation(self,dataframe,features,target,product,max_depth=None,min_samples_leaf=None):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)


		if max_depth is not None:
			self.max_depth = max_depth

		if min_samples_leaf is not None:
			self.min_samples_leaf = min_samples_leaf

		products = dataframe[product].unique()
		

		models = {}

		new_df = pd.DataFrame()

		for product in products:

			print(product)

			sub = dataframe[dataframe['Product_Group']==product]
			
			models[product] = {}
			
			clf = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=self.random_state, ccp_alpha=self.ccp_alpha)
			clf.fit(sub[features],sub[target])

			filename = utils.get_unique_filename('clf')
			filename = '_'+filename+'.py'			
			
			models[product] = filename

			res = self.export_py_code(clf, feature_names=features, spacing=4)
	
			# save the string as a function using above filename
			with open(filename, 'w') as f:
				f.write(res)    
			
			module = import_file(filename)
			
			# # rename column names
			for a,b in zip(features,utils.get_func_args(module.decision_tree)): 
				sub.rename(columns={a:b},inplace=True)
			
			sub['node'] = sub[utils.get_func_args(module.decision_tree)].apply(lambda x: module.decision_tree(*x), axis=1)
			sub['node'] = sub['node'].apply(lambda x:str(x).replace('[','').replace(']',''))

			# remove the file
			os.remove(filename)

			# encode the segment
			le = LabelEncoder()
			sub['segment'] = le.fit_transform(sub['node'])

			new_df = new_df.append(sub)

		return new_df, 'segment'

	def lin_reg(self,X,y):

		X = np.array(X)
		y = np.array(y)

		lin_reg = LinearRegression()
		lin_reg.fit(X.reshape(-1, 1),y.reshape(-1, 1))
		
		return lin_reg.intercept_, lin_reg.coef_
	

	def price_elasticity(self,data,segment_column,product_column,target_volume,target_price):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)
		print(msg)

		#empty segment_column means B2C
		# print(segment_column)		

		products = data[product_column].unique()

		# print(products)

		new_df = pd.DataFrame()

		if segment_column=='':

			#B2C

			for product in products:
	
				sub = data[data[product_column]==product]			    			    			

				X = sub[target_price]    
				y = sub[target_volume]            
				
				intercept, coef = self.lin_reg(X,y)
				
				intercept = intercept[0]        
				coef = coef[0][0]
				
				sub.loc[:,'elasticity_'] = coef
							
				new_df = new_df.append(sub)
	

		else:

			for product in products:
	
				sub = data[data[product_column]==product]
				segments = sub[segment_column].unique()
					
				for segment in segments:
			  
					sub_segment = sub[sub[segment_column]==segment]        
					
					X = sub_segment[target_price]
					y = sub_segment[target_volume]
					
					intercept, coef = self.lin_reg(X,y)
				
					intercept = intercept[0]        
					coef = coef[0][0]
					
					sub_segment.loc[:,'elasticity_'] = coef
								
					new_df = new_df.append(sub_segment)
		
		return new_df

	def opt(self, current_price, sales_volume, standard_cost, price_elasticity, original_pinc, with_constraint=False,volume_constrain=None):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()	
		self.log.print_(msg)
		print(msg)

		if volume_constrain is not None:

			self.volume_constrain = volume_constrain

		# print('constraint:',with_constraint)
		
		revenue_calc = current_price*sales_volume
		
		def f(x):
		
			original_pinc = x[0]
		
			optimal_price = (1+original_pinc)*current_price
			new_volume = sales_volume*(1+(price_elasticity*original_pinc))
			volume_change = new_volume/sales_volume-1
			new_revenue = optimal_price*new_volume
			delta_revenue = new_revenue-revenue_calc
			new_margin = ((new_revenue/new_volume)-standard_cost)*new_volume
			old_margin = (current_price-standard_cost)*sales_volume
			delta_margin = new_margin-old_margin

			return delta_margin


		def calculatevolume_change(x):
			
			original_pinc = x[0]
		
			optimal_price = (1+original_pinc)*current_price
			new_volume = sales_volume*(1+(price_elasticity*original_pinc))
			volume_change = new_volume/sales_volume-1
			new_revenue = optimal_price*new_volume
			delta_revenue = new_revenue-revenue_calc
			new_margin = ((new_revenue/new_volume)-standard_cost)*new_volume
			old_margin = (current_price-standard_cost)*sales_volume
			delta_margin = new_margin-old_margin

			return new_volume/sales_volume-1

		def objective(x):
			# minus sign means the opposite of minimize
			return -f(x)

		def constraint(x):
			return calculatevolume_change(x)+self.volume_constrain
		
		cons = ({'type':'ineq','fun':constraint})
		
		x0 = np.array([original_pinc])

		if with_constraint:
			
			sol = minimize(objective,x0,constraints=cons,options={'disp':True})        
		
		else:
			
			sol = minimize(objective,x0,options={'disp':True})

		xOpt = sol.x
			
		print('xOpt:',xOpt.round(2),len(xOpt),type(xOpt))
		
		if len(xOpt)==1 and isinstance(xOpt, np.ndarray):     
			
			return xOpt[0]
		
		else:
		
			return 0


	def price_optimisation(self,type_,filepath,features,price_column,volume_column,product_column,current_price,sales_volume,standard_cost,segmentation_features):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()	
		self.log.print_(msg)
		print(msg)

		return_ = dict()		

		status = 0	
		error = None

		
		if isinstance(type_, str): 


			if type_!='B2B' and type_!='B2C':


				msg = 'Wrong value for type parameter'
				self.log.print_(msg)
				print(msg)

				error = msg

			else:


				# check filepath type
				if isinstance(filepath, str): 

					#check if file is a csv file

					if filepath.split('.')[-1]=='csv':

						#check features type
						if isinstance(features, list):
							
							try:
								data = pd.read_csv(filepath)
							except Exception as e:

								msg = e
								self.log.print_(msg)
								print(msg)
								
								error = str(msg)

							else:


								#check if features exist in dataframe
								if all(elem in list(data.columns) for elem in features):
									
									#check price_column type
									if isinstance(price_column, str):

										#check if price_column exists in dataframe
										if price_column in list(data.columns):

											#check volume_column type
											if isinstance(volume_column, str):

												#check if price_column exists in dataframe
												if volume_column in list(data.columns):

													#check product column type
													if isinstance(product_column, str):

														#check if price_column exists in dataframe
														if product_column in list(data.columns):

															#check product column type
															if isinstance(current_price, str):

																#check if current_price exists in dataframe
																if current_price in list(data.columns):

																	#check product column type
																	if isinstance(sales_volume, str):

																		#check if current_price exists in dataframe
																		if sales_volume in list(data.columns):

																			#check product column type
																			if isinstance(standard_cost, str):

																				#check if current_price exists in dataframe
																				if standard_cost in list(data.columns):

																					
																					#check if features exist in dataframe
																					if all(elem in list(data.columns) for elem in segmentation_features):
					
																						print(type_)

																						#clean all numerical features
																						categorical_features, numerical_features = self.group_features(data,features)

																						print(categorical_features)
																						print(numerical_features)

																						#feature engineering
																						data = self.target_encoding(data,categorical_features,price_column)

																						try:

																							segment_column = ''

																							#if it B2B then create a segmentation first
																							if type_=='B2B':

																								data, segment_column = self.segmentation(data,segmentation_features,price_column,product_column)

																							

																							#anticipate zero values, if the target and volume have zero values  
																							target_volume = 'target_volume'
																							target_price = 'target_price'

																							# if type_=='B2C':

																							for col in [volume_column]+[price_column]:
																								
																								if data[data[volume_column]==0].shape[0]:        
																									data[target_volume] = data[volume_column]    
																								else:
																									data[target_volume] = np.log(data[volume_column])
																									
																								if data[data[price_column]==0].shape[0]:        
																									data[target_price] = data[price_column]    
																								else:
																									data[target_price] = np.log(data[price_column])


																							#price elasticity
																							# print(data.shape)
																							data = self.price_elasticity(data,segment_column,product_column,target_volume,target_price)
																							# print(data.shape)

																							# print(data['elasticity_'].unique())
																							
																							data['AVG_PE'] = data.groupby(product_column)['elasticity_'].transform('mean')																					
																							data['elasticity_'] = np.where(data['elasticity_']>=self.elasticity_cutoff,data['AVG_PE'],data['elasticity_'])	

																							# print(data['elasticity_'].unique())																					

																							#optimisation
																							data['original_pinc'] = 0
																							data['with_constraints'] = True

																							# print(data)
																							# data = data.head(5)

																							# opt(self, current_price, sales_volume, standard_cost, price_elasticity, original_pinc, with_constraint=False):																				
																							data['pinc_'] = data[[current_price]+[sales_volume]+[standard_cost]+['elasticity_']+['original_pinc']].apply(lambda x:self.opt(*x), axis=1)
																							data['pinc_const_'] = data[[current_price]+[sales_volume]+[standard_cost]+['elasticity_']+['original_pinc']+['with_constraints']].apply(lambda x:self.opt(*x), axis=1)

																							# print(data)
																							# data.to_csv(type_+'_final.csv',index=False)

																							out = data.to_json(orient='records')
																							# result = json.dumps(out)
																							result = str(out)
																							status = 1

																						except Exception as e:

																							msg = e
																							self.log.print_(msg)
																							print(msg)

																							error = msg
																					else:

																						msg = 'One or all segmentation features does not exist in dataframe'
																						self.log.print_(msg)
																						print(msg)

																						error = msg

																				else:

																					msg = standard_cost+' does not exist in dataframe'
																					self.log.print_(msg)
																					print(msg)

																					error = msg


																			else:

																				msg = 'Wrong type of parameter for standard_cost'
																				self.log.print_(msg)
																				print(msg)

																				error = msg



																		else:

																			msg = sales_volume+' does not exist in dataframe'
																			self.log.print_(msg)
																			print(msg)

																			error = msg


																	else:

																		msg = 'Wrong type of parameter for sales_volume'
																		self.log.print_(msg)
																		print(msg)

																		error = msg


																else:

																	msg = current_price+' does not exist in dataframe'
																	self.log.print_(msg)
																	print(msg)

																	error = msg

															else:


																msg = 'Wrong type of parameter for current_price'
																self.log.print_(msg)
																print(msg)

																error = msg


														else:

															msg = product_column+' does not exist in dataframe'
															self.log.print_(msg)
															print(msg)

															error = msg


													else:

														msg = 'Wrong type of parameter for volume_column'
														self.log.print_(msg)
														print(msg)

														error = msg


												else:

													msg = volume_column+' does not exist in dataframe'
													self.log.print_(msg)
													print(msg)

													error = msg


											else:

												msg = 'Wrong type of parameter for volume_column'
												self.log.print_(msg)
												print(msg)

												error = msg


										else:

											msg = price_column+' does not exist in dataframe'
											self.log.print_(msg)
											print(msg)

											error = msg

									else:

										msg = 'Wrong type of parameter for price_column'
										self.log.print_(msg)
										print(msg)

										error = msg



								else:

									msg = 'One or all features does not exist in dataframe'
									self.log.print_(msg)
									print(msg)

									error = msg

						else:

							msg = 'Features should be a list'
							self.log.print_(msg)
							print(msg)

							error = msg

					else:

						msg = 'Filename extension should be .csv'
						self.log.print_(msg)
						print(msg)

						error = msg



				else:

					msg = 'Wrong parameter for filepath'
					self.log.print_(msg)
					print(msg)

					error = msg


		else:

			msg = 'Wrong type of parameter for type'
			self.log.print_(msg)
			print(msg)

			error = msg



		

		return_["status"] = status
		return_["error"] = error

		if status==1:
			return_["data"] = result
		else:
			return_["data"] = None		

		return_json = json.dumps(return_)

		return return_json





	# def status_check(self,filepath:string,features:list, target, index=None):

	# 	msg = self.__class__.__name__+'.'+utils.get_function_caller()
	# 	self.log.print_(msg)
	# 	print(msg)

	# 	status = 0
	# 	result = None
	# 	error = None

	# 	# check if the file is csv file
	# 	try:

	# 		ext = filepath.split('.')[-1]			

	# 		if ext.lower()=='csv':

	# 			msg = 'File is a csv file'
	# 			self.log.print_(msg)
	# 			print(msg)

	# 			status = 1

	# 		else:

	# 			msg = 'File is not a csv file'
	# 			self.log.print_(msg)
	# 			print(msg)

	# 			error = msg

	# 			return status, error

	# 	except Exception as e:

	# 		msg = 'Error when checking the file extension'
	# 		self.log.print_(msg)
	# 		print(msg)


	# 	if status:

	# 		# check if the all features in the file
	# 		try:

	# 			try:

	# 				data = pd.read_csv(filepath)
					
	# 			except Exception as e:

	# 				msg = 'Cannot read the input file'
	# 				self.log.print_(msg)
	# 				print(msg)

	# 				error = msg

	# 				return status, error

	# 			all_features = data.columns

	# 			# # dataframe size should be >1
	# 			if data.shape[0]:

	# 				msg = 'Dataframe shape:',data.shape
	# 				self.log.print_(msg)
	# 				print(msg)

	# 			else:

	# 				msg = 'DataFrame has no data'
	# 				self.log.print_(msg)
	# 				print(msg)

	# 				status = 0
	# 				error = msg

	# 				return status, error


	# 			# check if index exists and unique
	# 			if index is not None:

	# 				if index in all_features:

	# 					msg = 'Index exists'
	# 					self.log.print_(msg)
	# 					print(msg)

	# 				else: 
						
	# 					msg = 'Index does not exist'
	# 					self.log.print_(msg)
	# 					print(msg)

	# 					status = 0
	# 					error = msg

	# 					return status, error

	# 				if not data[index].is_unique:

	# 					msg = 'Index does not have unique values'
	# 					self.log.print_(msg)
	# 					print(msg)

	# 					status = 0
	# 					error = msg

	# 					return status, error
					

	# 			# check if target exists and type is float
	# 			# print(target)
	# 			# print(all_features)
	# 			if target in all_features:

	# 				msg = 'Target feature exists'
	# 				self.log.print_(msg)
	# 				print(msg)

	# 			else:

	# 				msg = 'Target feature does not exist'
	# 				self.log.print_(msg)
	# 				print(msg)

	# 				status = 0
	# 				error = msg

	# 				return status, error
				
	# 			if 'float' in str(data[target].dtype):

	# 				msg = 'Target feature type is acceptable'
	# 				self.log.print_(msg)
	# 				print(msg)

	# 			else:

	# 				msg = 'Target feature type is not acceptable'
	# 				self.log.print_(msg)
	# 				print(msg)

	# 				status = 0
	# 				error = msg

	# 				return status, error
					
	# 			# check if features sub of all_features
	# 			if not set(features).issubset(set(all_features)):

	# 				msg = 'All or some of the features do not exists'
	# 				self.log.print_(msg)
	# 				print(msg)

	# 				status = 0
	# 				error = msg

	# 				return status, error
				
	# 			status = 1

	# 			return status, error
				

	# 		except Exception as e:

	# 			msg = 'Error when checking the features, target feature and index'
	# 			self.log.print_(msg)
	# 			print(msg)

	# 			status = 0

	# 	return status		


	# def feature_engineering(self,dataframe,features,target):

	# 	msg = self.__class__.__name__+'.'+utils.get_function_caller()
	# 	self.log.print_(msg)
	# 	print(msg)

	# 	categorical_features = []
	# 	numerical_features = []

	# 	for col,type_ in zip(dataframe[features].columns,dataframe[features].dtypes):
			
	# 		if str(type_)=='object' and col!=target:        
	# 			categorical_features.append(col)        
	# 		else:
	# 			numerical_features.append(col)

	# 	# handle empty values
	# 	for categorical_feature in categorical_features:

	# 		dataframe[categorical_feature].fillna('No data', inplace=True)

	# 	for numerical_feature in numerical_features:

	# 		dataframe[numerical_feature].fillna(dataframe[numerical_feature].median(), inplace=True)


	# 	# feature engineering
	# 	for categorical_feature in categorical_features:

	# 		dataframe[categorical_feature] = dataframe.groupby(categorical_feature)[target].transform('max')

	# 	return dataframe
			

	# def features_assessment(self,filepath:string,features:list,target:string):

	# 	msg = self.__class__.__name__+'.'+utils.get_function_caller()
	# 	self.log.print_(msg)
	# 	print(msg)

	# 	# print('target:',target)

	# 	return_ = dict()		

	# 	if isinstance(filepath, str) and isinstance(features, list) and isinstance(target, str):

	# 		target = target.strip()

	# 		status, error = self.status_check(filepath,features,target)

	# 		if status :

	# 			try:

	# 				data = pd.read_csv(filepath)

	# 				# status, target = self.check_target(target)

	# 				if status:

	# 					# handle outliers
	# 					data = data[
	# 								(
	# 									(data[target]>=(data[target].mean() - (3*data[target].std())))
	# 									& (data[target]<=(data[target].mean() + (3*data[target].std())))
	# 								)
	# 								]

	# 					# do the feature assement

	# 					data = self.feature_engineering(data,features,target)

	# 					if len(features)==1:

	# 						freg = f_regression(np.array(data[features].values.reshape(-1, 1),dtype=float), np.array(data[target].values.ravel(),dtype=float))

	# 					else:

	# 						freg = f_regression(np.array(data[features].values,dtype=float), np.array(data[target].values.ravel(),dtype=float))

	# 					p_values = freg[1]

	# 					feature_pval = dict()

	# 					for p_value, feature in zip(p_values,features):
				
	# 						feature_pval[feature] = round(p_value, 3)    

						
	# 					result = json.dumps(feature_pval)

	# 				else:

	# 					msg = 'Error on target feature'
	# 					self.log.print_(msg)
	# 					print(msg)

	# 					status = 0	
	# 					result = None



	# 			except:

	# 				msg = 'Error when doing features_assessment'
	# 				self.log.print_(msg)
	# 				print(msg)

	# 				status = 0	
	# 				result = None



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

	# 	return_json = json.dumps(return_)

	# 	return return_json

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

	# def get_func_args(self,f):

	# 	if hasattr(f, 'args'):
	# 		return f.args
	# 	else:
	# 		return list(inspect.signature(f).parameters)


	# def segmentation(self,filepath:string,features:list,target:string, index:string, max_depth=None, min_samples_leaf=None):

	# 	msg = self.__class__.__name__+'.'+utils.get_function_caller()
	# 	self.log.print_(msg)
	# 	print(msg)
		
	# 	# tree
	# 	if max_depth is not None:
	# 		self.max_depth = max_depth

	# 	if min_samples_leaf is not None:
	# 		self.min_samples_leaf = min_samples_leaf


	# 	return_ = dict()

	# 	if isinstance(filepath, str) and isinstance(features, list) and isinstance(target, str) or isinstance(filepath, str) and isinstance(features, list) and isinstance(target, str) and isinstance(index, str):

	# 		target = target.strip()

	# 		if index is not None:
	# 			index = index.strip()

	# 		status, error = self.status_check(filepath,features,target,index)		

	# 		if status:

	# 			try:				

	# 				data = pd.read_csv(filepath)
					
	# 				# handle outliers		
	# 				data = data[
	# 							(
	# 								(data[target]>=(data[target].mean() - (3*data[target].std())))
	# 								& (data[target]<=(data[target].mean() + (3*data[target].std())))
	# 							)
	# 							]

	# 				data = self.feature_engineering(data,features,target)

	# 				clf = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=self.random_state, ccp_alpha=self.ccp_alpha)
	# 				clf.fit(data[features],data[target])		

	# 				# get the tree rules
	# 				res = self.export_py_code(clf, feature_names=features, max_depth=self.max_depth, spacing=4)					

	# 				# save the function as a file
	# 				filename = '_'+utils.get_unique_filename('tree')+'.py'
					
	# 				with open(filename, 'w') as f:
	# 					f.write(res)

	# 				module = import_file(filename)					
					
	# 				# rename column names
	# 				for a,b in zip(features,self.get_func_args(module.decision_tree)): 
	# 					data.rename(columns={a:b},inplace=True)


	# 				data['node'] = data[self.get_func_args(module.decision_tree)].apply(lambda x: module.decision_tree(*x), axis=1)
	# 				data['node'] = data['node'].apply(lambda x:str(x).replace('[','').replace(']',''))
					
	# 				# remove the file
	# 				os.remove(filename)

	# 				# encode the segment
	# 				le = LabelEncoder()
	# 				data['segment'] = le.fit_transform(data['node'])


	# 				output_df = data[[index]+['segment']]
	# 				output_df = output_df.groupby(['segment'])[index].apply(list).reset_index() 
					

	# 				# convert dataframe to json
	# 				# output_json = output_df.to_json(orient ='records')
	# 				outputs = []

	# 				for i, rows in enumerate(output_df.itertuples(),1):

	# 					_str = '{"segment":'+str(rows[1])+', "'+index+'":'+str(rows[2])+'}'
	# 					outputs.append(_str)

	# 				final_output = '['+', '.join(outputs)+']'
	# 				final_output = final_output.replace("'",'"')					
	# 				output_json = json.loads(final_output)



	# 			except Exception as e:

	# 				msg = 'Error on modeling'
	# 				self.log.print_(msg)
	# 				print(msg)

	# 				print(e)

				
	# 		else:

	# 			msg = 'There is problem on the file path, features, target feature, or index'
	# 			self.log.print_(msg)
	# 			print(msg)

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

	# 		return_["data"] = output_json

	# 	else:

	# 		return_["data"] = None			

	# 	return_ = json.dumps(return_)

	# 	# # save this json for testing purpose
	# 	# with open('sample_json.json', 'w') as f:
	# 	# 	f.write(return_)

	# 	return return_


	# def get_threshold(self,price_threshold,is_power_index,segment_field,segment=None):

	# 	msg = self.__class__.__name__+'.'+utils.get_function_caller()
	# 	self.log.print_(msg)
	# 	print(msg)

	# 	try :

	# 		if (is_power_index==True and isinstance(price_threshold, list)):

	# 			threshold = [threshold for threshold in price_threshold if threshold[segment_field]==segment]

	# 			if len(threshold)==1:
					
	# 				threshold = threshold[0]['threshold'][0]
	# 				return threshold

	# 			else:
	# 				raise Exception('There is an error in the JSON threshold file')



	# 		elif (is_power_index==False and isinstance(price_threshold, dict)):

	# 			return price_threshold



	# 	except Exception as e:

	# 		msg = 'There is an error in the JSON threshold file'
	# 		self.log.print_(msg)
	# 		print(msg)

	# 		status = 0	
	# 		error = msg
	# 		result = None							

	# def read_json(self,json_file):

	# 	msg = self.__class__.__name__+'.'+utils.get_function_caller()
	# 	self.log.print_(msg)
	# 	print(msg)

	# 	if isinstance(json_file, str):

	# 		try:

	# 			if self.env == 'local':

	# 				with open(json_file) as f:
	# 					json_output = json.load(f)

	# 			elif self.env == 'prod':

	# 				resp = requests.get(json_file)
	# 				json_output = json.loads(resp.text)   

	# 		except Exception as e:

	# 			msg = 'There is an error when reading JSON file'
	# 			self.log.print_(msg)
	# 			print(msg)

	# 		finally:
				
	# 			return json_output

	# def get_threshold(self, threshold_item, key):

	# 	msg = self.__class__.__name__+'.'+utils.get_function_caller()
	# 	self.log.print_(msg)
	# 	print(msg)

	# 	try:

	# 		if isinstance(threshold_item, dict) and isinstance(key, str):

	# 			return threshold_item[key]

	# 	except Exception as e:

	# 		msg = 'There is problem with the threshold item'
	# 		self.log.print_(msg)
	# 		print(msg)

	# def get_price_per_segment(self,price_per_segment, segment, target):

	# 	msg = self.__class__.__name__+'.'+utils.get_function_caller()
	# 	self.log.print_(msg)
	# 	print(msg)

	# 	try:

	# 		if isinstance(price_per_segment, list) and (isinstance(target,str)):

	# 			return price_per_segment[segment][target]

	# 	except Exception as e:

	# 		msg = 'There is problem with the threshold item'
	# 		self.log.print_(msg)
	# 		print(msg)

	# def segment_check(self,price_per_segment,threshold,segment):

	# 	msg = self.__class__.__name__+'.'+utils.get_function_caller()
	# 	self.log.print_(msg)
	# 	print(msg)

	# 	A = [item[segment] for item in price_per_segment]					
	# 	B = [list(utils.find(segment, item))[0] for item in threshold ]

	# 	if len(A)==len(B) and set(A) == set(B):

	# 		return True

	# 	else:

	# 		return False



	# def price_segmentation(self, price_per_segment, threshold, segment, target):

	# 	msg = self.__class__.__name__+'.'+utils.get_function_caller()
	# 	self.log.print_(msg)
	# 	print(msg)

	# 	return_ = dict()

	# 	status = 0
	# 	error = None

	# 	if isinstance(price_per_segment, str) and isinstance(threshold, str) and isinstance(segment, str) and isinstance(target, str):

	# 		price_per_segment = price_per_segment.strip()
	# 		threshold = threshold.strip()
	# 		segment = segment.strip()
	# 		target = target.strip()

	# 		# check if number of segments in price per segment = number of segments in threshold

	# 		# read json 				
	# 		try:

	# 			price_per_segment_json = self.read_json(price_per_segment)
	# 			threshold_json = self.read_json(threshold)							

	# 		except Exception as e:

	# 			msg = 'Error on JSON file'
	# 			self.log.print_(msg)
	# 			print(msg)

	# 			print(e)

	# 			status = 0 
	# 			error = msg

	# 		else:

	# 			# print(price_per_segment_json)				
	# 			price_per_segment_df = pd.DataFrame(price_per_segment_json)
	# 			# print(price_per_segment_df)

	# 			if (segment in price_per_segment_df.columns):
				
	# 				if target in price_per_segment_df.columns:
				
	# 					# check json level of depth
	# 					depth = utils.depth(threshold_json)

	# 					segments = price_per_segment_df[segment].unique()

	# 					try:

	# 						if not segment in threshold_json and depth==1:

	# 							output = list()
	# 							for i,segment_ in enumerate(segments):

	# 								line = list()    
	# 								line.append('"'+segment+'":'+str(segment_))
																
	# 								prices = self.get_price_per_segment(price_per_segment_json, segment_,target)

	# 								_str = []
	# 								for k,v in threshold_json.items():
										
	# 									tmp = '{"'+str(k)+'":'+str(np.percentile(prices, v*10))+'}'        
	# 									_str.append(tmp)
																		
	# 								# _str = '"'+target+'":['+','.join(_str)+']'								
	# 								_str = '"threshold":['+','.join(_str)+']'								

	# 								line.append(_str)
	# 								line = ','.join(line)								
	# 								line = "{"+line+"}"
									
	# 								output.append(line)
								
	# 							output = '['+','.join(output)+']'

	# 							result = json.loads(output)

	# 							# print(result)


	# 							status = 1


	# 						else:


	# 							if self.segment_check(price_per_segment_json,threshold_json,segment) :


	# 								# print('depth:',depth)
	# 								# print(threshold_json)
	# 								threshold_json_str = str(threshold_json)

	# 								# print('Original:',threshold_json_str)

	# 								# result = []

	# 								for i, segment_ in enumerate(segments):

	# 									prices = self.get_price_per_segment(price_per_segment_json, segment_,target)

	# 									item = threshold_json[segment_]
	# 									item_str = str(item)
	# 									new_item_str = str(item)

	# 									# print(type(item))
	# 									# print('Original item:',item)
	# 									# print('item_str:',item_str)
										
	# 									thresholds = list(utils.find('threshold', item))
	# 									# print(len(thresholds),thresholds)
										
	# 									# print('\n\n\n')

	# 									for threshold in thresholds:

	# 										threshold_ = threshold[0]		
	# 										original_threshold_str = str(threshold_)

	# 										# print('original_threshold_str:',original_threshold_str)							

	# 										for k,v in threshold_.items():

	# 											threshold_.update({k: np.percentile(prices, v*10)})

	# 										# print('new values: ',str(threshold_))
	# 										# print('Replace: ',original_threshold_str,'with',str(threshold_),'on',item_str)

	# 										# update the original item using string operation
	# 										new_item_str = new_item_str.replace(original_threshold_str,str(threshold_))

	# 										# print('Before replacement:',item_str)
	# 										# print('After replacement:',new_item_str)



	# 										# result.append(item_str)
	# 										# print('\n\n\n')


	# 									# print('replace',item_str,'with',new_item_str)
	# 									threshold_json_str = threshold_json_str.replace(item_str,new_item_str) 



	# 									# print('***************************************************')
	# 								# print('final:',threshold_json_str)

	# 								# # result = json.loads(result)
	# 								# result = '['+','.join(threshold_json_str).replace("'",'"')+']'
	# 								result = threshold_json_str.replace("'",'"')
	# 								result = json.loads(result)
	# 								# print(result)
	# 								status = 1

	# 							else:

	# 								msg = 'Segment are not same'
	# 								self.log.print_(msg)
	# 								print(msg)

	# 								status = 0	
	# 								error = msg
	# 								result = None
	# 					except Exception as e:

	# 						msg = 'Error when do price segmentation'
	# 						self.log.print_(msg)
	# 						print(msg)

	# 						print(e)

	# 						status = 0	
	# 						error = msg
	# 						result = None




	# 				else:

	# 						msg = 'Target does not exist'
	# 						self.log.print_(msg)
	# 						print(msg)

	# 						status = 0	
	# 						error = msg
	# 						result = None

	# 			else:

	# 					msg = 'Segment does not exist'
	# 					self.log.print_(msg)
	# 					print(msg)

	# 					status = 0	
	# 					error = msg
	# 					result = None

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







				
	# # def price_segmentation(self, price_per_segment, price_threshold, segment, target, is_power_index=False):

	# # 	msg = self.__class__.__name__+'.'+utils.get_function_caller()
	# # 	self.log.print_(msg)
	# # 	print(msg)

	# # 	return_ = dict()

	# # 	status = 0
	# # 	error = None

	# # 	if isinstance(price_per_segment, str) and isinstance(price_threshold, str) and isinstance(is_power_index, bool):
			
	# # 		try:

	# # 			price_per_segment = price_per_segment.strip()
	# # 			price_threshold = price_threshold.strip()
				

	# # 			# read json 				
	# # 			try:

	# # 				price_per_segment_json = self.read_json(price_per_segment)
	# # 				price_threshold_json = self.read_json(price_threshold)
				

	# # 			except Exception as e:

	# # 				msg = 'Error on JSON file'
	# # 				self.log.print_(msg)
	# # 				print(msg)

	# # 				print(e)

	# # 				status = 0 
	# # 				error = msg

	# # 			else:


	# # 				if (is_power_index==False and isinstance(price_threshold_json, dict)) or (is_power_index==True and isinstance(price_threshold_json, list)):

	# # 					price_per_segment_df = pd.DataFrame(price_per_segment_json)

	# # 					# check if segment and target exists
	# # 					if segment in price_per_segment_df.columns:

	# # 						if target in price_per_segment_df.columns:


	# # 							output = list()
	# # 							for i,segment_ in enumerate(price_per_segment_df[segment].unique()):

	# # 								line = list()    
	# # 								line.append('"'+segment+'":'+str(segment_))
									  
	# # 								sub = price_per_segment_df[price_per_segment_df[segment]==segment_]

	# # 								prices = sub[target].values[0]
													

	# # 								if is_power_index:
	# # 									threshold = self.get_threshold(price_threshold_json,is_power_index,segment_field=segment, segment=segment_)
	# # 								else:
	# # 									threshold = self.get_threshold(price_threshold_json,is_power_index,segment_field=segment)



	# # 								_str = []
	# # 								for k,v in threshold.items():
										
	# # 									tmp = '{"'+str(k)+'":'+str(np.percentile(prices, v*10))+'}'        
	# # 									_str.append(tmp)
										
									
	# # 								_str = '"'+target+'":['+','.join(_str)+']'
									

	# # 								line.append(_str)
	# # 								line = ','.join(line)
	# # 								line = "{"+line+"}"
									
									
	# # 								output.append(line)					    
									
	# # 							output = '['+','.join(output)+']'

	# # 							result = json.loads(output)

	# # 							status = 1



	# # 						else:

	# # 							msg = 'Target does not exist'
	# # 							self.log.print_(msg)
	# # 							print(msg)

	# # 							status = 0	
	# # 							error = msg
	# # 							result = None
	# # 					else:

	# # 						msg = 'There is an error in the JSON threshold file'
	# # 						self.log.print_(msg)
	# # 						print(msg)

	# # 						status = 0	
	# # 						error = msg
	# # 						result = None

	# # 				else:

	# # 					msg = 'Segment does not exist'
	# # 					self.log.print_(msg)
	# # 					print(msg)

	# # 					status = 0	
	# # 					error = msg
	# # 					result = None

					
	# # 		except Exception as e:

	# # 			msg = 'Error when do price segmentation!!'
	# # 			self.log.print_(msg)
	# # 			print(msg)

	# # 			print(e)


			




	# # 	else:

	# # 		msg = 'Parameters type are wrong'
	# # 		self.log.print_(msg)
	# # 		print(msg)

	# # 		status = 0	
	# # 		error = msg
	# # 		result = None


	# # 	return_["status"] = status
	# # 	return_["error"] = error

	# # 	if status==1:

	# # 		return_["data"] = result

	# # 	else:

	# # 		return_["data"] = None			

	# # 	return_ = json.dumps(return_)

	# # 	return return_
