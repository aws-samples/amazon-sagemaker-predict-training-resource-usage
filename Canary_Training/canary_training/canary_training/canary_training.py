import os
import json
import pandas
import numpy
#import matplotlib
#from matplotlib import pyplot
import scipy
import logging
import sklearn
import boto3
import random
from time import gmtime, strftime
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures 
logger = logging.getLogger('log')
import os
import boto3
import copy
import re
import sagemaker
from time import gmtime, strftime
import sagemaker
import boto3
from sagemaker import image_uris
from sagemaker.session import Session
from sagemaker.inputs import TrainingInput
from functools import partial
#import matplotlib
#set logs if not done already
if not logger.handlers:
	logger.setLevel(logging.INFO)
import os
import json
import pandas
import numpy
#import matplotlib
#from matplotlib import pyplot
import scipy
import logging
import sklearn
import math
import copy
from random import choices
import boto3
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures 
logger = logging.getLogger('log')
#set logs if not done already
if not logger.handlers:
	logger.setLevel(logging.INFO)

import sagemaker
import boto3
class CanaryTrainingKickoff:
		def __init__(self,instance_types=None,training_percentages=None,data_location=None,output_s3_location=None,the_temp_dir=None):
			default_instance_types=["ml.m5.large","ml.m5.xlarge","ml.m5.2xlarge"]
			default_percentages=[.01,.02,.03]
			region = boto3.Session().region_name
			default_dir=f"canary-training-{strftime('%Y-%m-%d-%H-%M-%S', gmtime())}-{str(random.random())}".replace(".","")
			default_output_s3_location=f'''s3://{sagemaker.Session().default_bucket()}/{default_dir}'''
			if output_s3_location==None:
				self.output_s3_location=default_output_s3_location
				#print(self.output_s3_location)
			else:
				self.output_s3_location=output_s3_location
				
			if instance_types==None:
				self.instance_types=default_instance_types
			else:
				self.instance_types=instance_types
				
			if training_percentages==None:
				self.training_percentages=default_percentages
			else:
				self.training_percentages=training_percentages
			self.data_location=data_location
			#create a temp dir
			if the_temp_dir==None:
				self.temp_dir=f"canary-training-temp-dir-{str(random.random())}".replace(".","")
			else:
				self.temp_dir=the_temp_dir
			os.system(f'mkdir {self.temp_dir}')
		
		def parse_s3_uri(self,s3_uri=None):
			bucket=re.search("^s3://(.*?)/",s3_uri).group(1)
			key=re.search(f'''^s3://{bucket}/(.*)''',s3_uri).group(1)
			return(bucket,key)        
		#def copy_directory_to_s3(self,the_data=None,s3_uri=None):
		#	s3 = boto3.resource('s3')
		#	bucket,key=self.parse_s3_uri(s3_uri=s3_uri)
		#	the_object = s3.Object(bucket, key)
		#	s3.Bucket(bucket).upload_fileobj(the_data,key)
		def list_data_location(self,s3_uri=None):
			the_cmd='''aws s3 ls --recursive s3_uri/|awk '{print $4}' '''
			the_cmd=the_cmd.replace('s3_uri',s3_uri)
			the_bucket=s3_uri.split("/")[2]
			#print(the_cmd)
			the_results=os.popen(the_cmd).readlines()
			#print(the_results)
			#print(the_bucket)
			all_uris=[f'''s3://{the_bucket}/{i.rstrip()}''' for i in the_results ]
			return(all_uris)
		def create_manifest_file(self,the_objects_list=None,file_name=None):
			os.system(f'mkdir {self.temp_dir}/manifests ')
			f_out=open(f"{self.temp_dir}/manifests/{file_name}", 'w')
			print('''[''',file=f_out,sep="\n")
			the_dir=os.path.dirname(the_objects_list[0]) #assume all object under same path as first object
			the_dir_manifest="{\"prefix\":\"the_dir/\"}" #add / so it knows it is a path
			the_dir_manifest=the_dir_manifest.replace('the_dir',the_dir)
			print(the_dir_manifest+ ",",file=f_out)
			for i in range(0,len(the_objects_list)):
				the_object_name=os.path.basename(the_objects_list[i])
				the_object_name=f'''"{the_object_name}"'''
				if i< (len(the_objects_list) -1):
					print(the_object_name + ",",file=f_out) #print comma at end unless last
				else:
					print(the_object_name ,file=f_out)        
			print(''']''',file=f_out,sep="\n")
			#print(*the_objects_list)
			f_out.close()
		def copy_directory_to_s3(self,the_directory=None):
			the_cmd=f'''aws s3 cp --recursive {the_directory} {self.output_s3_location}/{the_directory}/'''
			print(the_cmd)
			os.system(the_cmd)
			return()
		def sample_uris(self,the_uris=None):
			'''create samples of uris lists'''
			num_uris= len(the_uris)
			min_uris=10
			if  num_uris <min_uris:
				logger.error(f'The data set must have at least {min_uris} objects to do canary training')
				return()
			num_obj_to_sample= [numpy.floor(i * num_uris) for i in self.training_percentages]
			num_obj_to_sample=[int(i) for i in num_obj_to_sample ] #num objects are an integer
			the_sampled_lists=[random.sample(the_uris, n) for n in num_obj_to_sample]
			return(the_sampled_lists)
		def get_data_files(self):
			'''get a list of all files for training.'''
			#Only supports S3 for now.
			pass
		def delete_temp_dir(self):
			os.system(f'rm -rf {self.temp_dir}')
		def create_estimators(self,base_estimator=None):
			'''given an original estimator, create new estimators that use the manifest files '''
			new_estimators=[]
			for i in range(0,len(self.instance_types)):
				new_estimator=copy.copy(base_estimator)
				new_estimator.instance_type=self.instance_types[i]
				new_estimators.append(new_estimator)
			return(new_estimators)
		def create_manifests_for_uris(self,the_uris_list_list=None):
			for i in range(0,len(the_uris_list_list)):
				length_of_list=len(the_uris_list_list[i])
				self.create_manifest_file(the_objects_list=the_uris_list_list[i],file_name=f'the_manifest_{str(i)}.manifest')
			return()
		def create_manifest_training_inputs(self,*argv,**kwargs):
			'''*argv and **kwargs are passed to TrainingInput function'''
			manifest_dir=f'{self.output_s3_location}/{self.temp_dir}/manifests'
			manifest_locations=self.list_data_location(s3_uri=manifest_dir)
			manifest_training_input_objects=[sagemaker.inputs.TrainingInput(i,*args,s3_data_type='ManifestFile', **kwargs) for i in manifest_locations]
			return (manifest_training_input_objects)
		def copy_file_to_s3(self,the_file=None,s3_uri=None):
			the_cmd=f'''aws s3 cp {the_file} {s3_uri}'''
			print(the_cmd)
			os.system(the_cmd)
			return()
		def kick_off_canary_training(self,estimators_list=None,manifest_training_inputs_list=None,training_channels_list=['train'],
			estimator_args=None,
			estimator_kwarks=None):
		#now that we have the list of estimators, let's kick off the canary training jobs
		#how you execute the fit will depend on your specific ML workload
		#TODO, how to better handle passing options to fit? for other estimators
			all_jobs_l=[]
			#we need the total list of objects in the entire training
			the_uris=self.list_data_location(s3_uri=self.data_location)
			total_number_training_objects=len(the_uris)
			#tur n training list into a dict
			for i in estimators_list:
				for j in manifest_training_inputs_list:
					job_name=f"canary-training--job-{strftime('%Y-%m-%d-%H-%M-%S', gmtime())}-{str(random.random())}".replace(".","")
					#we assume that the training and validation sets are the same (presumambly small) size
					training_channels_dict={the_channel:j for the_channel in training_channels_list}
					#print(training_channels_dict)
					
					all_kwargs={"inputs":training_channels_dict,"job_name":job_name,"wait":False}
					if estimator_kwarks is not None:
						all_kwargs.update(estimator_kwarks) #update is in place
					if estimator_args is not None:
						i.fit(*estimator_args,**all_kwargs)
					else:
						i.fit(**all_kwargs)

					training_s3_uri=j.config['DataSource']['S3DataSource']['S3Uri']
					
					 
					
					
					#now we will get the number of files the manifest is pointing to
					s3 = boto3.resource('s3')
					bucket,key=self.parse_s3_uri(training_s3_uri)
					content_object = s3.Object(bucket, key)
					file_content = content_object.get()['Body'].read().decode('utf-8')
					json_content = json.loads(file_content)
					num_objects_in_manifest=len(json_content[1:])
					percentage_of_data_trained_on=num_objects_in_manifest/total_number_training_objects
	
					job_info=f'''{job_name},{i.instance_type},{j.config['DataSource']['S3DataSource']['S3Uri']},{str(percentage_of_data_trained_on)}'''
					all_jobs_l.append(job_info)
					
					
		

			print("Done Submitting Jobs")
			return(all_jobs_l)
		
		def copy_canary_training_job_info_to_s3(self,all_jobs_l=None):
			#save job names; this will allow you to run this second part without rerunning the entire set of estimators
			filename=f'{self.temp_dir}/data_files/canary_training_job_list.csv'
			os.makedirs(os.path.dirname(filename), exist_ok=True)
			outfile=open(filename,'w')
			print(*all_jobs_l,file=outfile,sep='\n')
			outfile.close()
			self.copy_file_to_s3(the_file=f'{self.temp_dir}/data_files/canary_training_job_list.csv',s3_uri=f'''{self.output_s3_location}/{self.temp_dir}/data_files/canary_training_job_list.csv''')
			self.canary_training_job_info=f'{self.output_s3_location}/{self.temp_dir}/data_files/canary_training_job_list.csv'
			return()
		def get_canary_training_jobs_list_from_temp_dir(self):
			the_file=f'{self.temp_dir}/data_files/canary_training_job_list.csv'
			job_info_l=open(the_file,'r').readlines()
			job_info_l=[i.rstrip() for i in job_info_l]
			return(job_info_l)
		def get_percentages_for_canary_training_jobs(self,submitted_jobs_information=None):
			'''for all the manifests created, get the percentage of the total data that they are.
			'''
			pass
		#We are getting the percentages this way and NOT relying on the options to avoid rounding errors.
		#    the_cmd=''

		#def canary_fit(self,estimators_list=None,manifests_lists=None,inputs=NOne):
		#    print(kwargs)
		#    print(args)
			
			
			
class GetJobProfileInfo:
		def __init__(self):
			#metrics to use from profiler.
			self.the_metrics_and_values={'CPUUtilization':[],
									"I/OWaitPercentage":[],
									"MemoryUsedPercent":[],
									'GPUUtilization':[],
									"GPUMemoryUtilization":[]
										}
									  
			pass
		def get_profiler_location_from_job(self,job_name):
			client = boto3.client('sagemaker')
			profile_log_location=client.describe_training_job(TrainingJobName=job_name)['ProfilerConfig']['S3OutputPath']
			profile_log_location=f"{profile_log_location}/{job_name}/profiler-output/system/incremental"
			return (profile_log_location)
		def get_profile_logs(self,s3_location=None):
			s3_uri=s3_location
			the_cmd='''aws s3 ls --recursive s3_location/|awk '{print $4}' |grep json'''
			the_cmd=the_cmd.replace('s3_location',s3_location)
			the_bucket=s3_uri.split("/")[2]
			#print(the_bucket)
			log_uris=[f'''s3://{the_bucket}/{i.rstrip()}''' for i in os.popen(the_cmd).readlines()]
			#print(log_uris)
			all_log_data_list=[]
			for the_file in log_uris:
				the_cmd=f'aws s3 cp {the_file} -'
				log_data=os.popen(the_cmd).readlines()
				for entry in log_data:
					try:
						all_log_data_list.append(entry.rstrip())
					except Exception as e:
						pass
			return(all_log_data_list)
		def get_metrics(self,the_logs=None):
			#see here for the definitions https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html
			the_metrics_and_values=self.the_metrics_and_values
			for i in the_logs:
				the_log=json.loads(i)
				try:
					the_log_name=the_log["Name"]
					the_log_type=the_log["Type"]
					the_log_dimension=the_log["Dimension"]
					if the_log_name=='MemoryUsedPercent':        
						the_metrics_and_values[the_log_name].append(the_log['Value'])
					elif the_log_type=='cpu' and the_log_dimension=='CPUUtilization':
						the_metrics_and_values[the_log_dimension].append(the_log['Value'])
					elif the_log_type=='cpu' and the_log_dimension=='I/OWaitPercentage':
						the_metrics_and_values[the_log_dimension].append(the_log['Value'])
					elif the_log_type=='gpu' and the_log_dimension=='GPUUtilization':
						the_metrics_and_values[the_log_dimension].append(the_log['Value'])
					elif the_log_type=='gpu' and the_log_dimension=='GPUMemoryUtilization':
						the_metrics_and_values[the_log_dimension].append(the_log['Value'])
				except Exception as e:
					pass

			return(the_metrics_and_values)

		def get_average_of_metrics(self,the_metrics_and_values=None):
			'''note, this may note be the actual average, it is just some kind of average or percentile'''
			dict_to_return={}
			the_keys=list(self.the_metrics_and_values.keys())
			for i in the_keys:
				#Try to get the mean; if it does not exist, then the metric was not calculated, and return -1 for that metric
				try:
					if len(the_metrics_and_values[i])>0: #set to -1 if the list exists but it is empty
						the_mean_value=numpy.percentile(the_metrics_and_values[i],99)
					else:
						the_mean_value=-1
				except:
					the_mean_value=-1
				dict_to_return[i]=the_mean_value
			return(dict_to_return)
		def normalize_by_num_cpus(self,input_metrics_dict=None,instance_type=None):
			'''normalize metrics by dividing CPUUtilization by number of CPUs in the instance type'''
			instance_num_cpus=self._get_instance_feature_data_cpu(instance_type=instance_type)
			#print(instance_num_cpus)
			input_metrics_dict['CPUUtilization']=input_metrics_dict['CPUUtilization']/instance_num_cpus
			return(input_metrics_dict)
		def normalize_by_num_gpus(self,input_metrics_dict=None,instance_type=None):
			'''normalize metrics by dividing CPUUtilization by number of CPUs in the instance type'''
			instance_num_gpus=self._get_instance_feature_data_gpu(instance_type=instance_type)
			#avoid divide by 0 error later by setting to 1 if value is 0
			if instance_num_gpus==0:
				instance_num_gpus=1
			#print(instance_num_cpus)
			input_metrics_dict['GPUUtilization']=input_metrics_dict['GPUUtilization']/instance_num_gpus
			return(input_metrics_dict)
		def _get_instance_feature_data_cpu(self,instance_type=None):
			package_directory = os.path.dirname(__file__)
			input_file="instance_data_unnormalized.csv"
			df_1=pandas.read_csv(os.path.join(package_directory,input_file))
			instance_num_cpus=df_1[df_1['instance_type']==instance_type]['vCPU']
			instance_num_cpus=int(instance_num_cpus)
			return(instance_num_cpus)
		def _get_instance_feature_data_gpu(self,instance_type=None):
			package_directory = os.path.dirname(__file__)
			input_file="instance_data_unnormalized.csv"
			df_1=pandas.read_csv(os.path.join(package_directory,input_file))
			instance_num_gpus=df_1[df_1['instance_type']==instance_type]['GPU']
			instance_num_gpus=int(instance_num_gpus)
			return(instance_num_gpus)
		def get_useful_jobs_info(self,submitted_jobs_information=None):
			client = boto3.client('sagemaker')
			all_useful_jobs_infos=[]
			for i in range(0,len(submitted_jobs_information)):
				basic_job_info_l=submitted_jobs_information[i].split(",")
				job_name=basic_job_info_l[0]
				raw_job_info=client.describe_training_job(TrainingJobName=job_name)
				useful_job_info={}
				#start getting important statistics from the job
				useful_job_info['TrainingJobStatus']=raw_job_info['TrainingJobStatus']
				useful_job_info['TrainingTimeInSeconds']=raw_job_info['TrainingTimeInSeconds']
				useful_job_info['InstanceType']=basic_job_info_l[1]
				useful_job_info["ManifestLocation"]=basic_job_info_l[2]
				useful_job_info['job_name']=job_name
				useful_job_info['PercentageDataTrainedOn']=basic_job_info_l[3]
				#print(job_name)


				#get profiler data
				#cjp=GetJobProfileInfo()
				s3_uri=self.get_profiler_location_from_job(job_name=job_name)
				#print(s3_uri)
				the_logs=self.get_profile_logs(s3_location=s3_uri)
				#print(the_logs)
				the_metrics_and_values=copy.deepcopy(self.get_metrics(the_logs=the_logs))
				#print(the_metrics_and_values)
				average_of_metrics=self.get_average_of_metrics(the_metrics_and_values=the_metrics_and_values)
				#print(average_of_metrics)
				instance_type=useful_job_info['InstanceType']
				average_of_metrics=self.normalize_by_num_cpus(input_metrics_dict=average_of_metrics,instance_type=instance_type)
				average_of_metrics=self.normalize_by_num_gpus(input_metrics_dict=average_of_metrics,instance_type=instance_type)
				#print(average_of_metrics)
				useful_job_info_2={**copy.deepcopy(useful_job_info),**copy.deepcopy(average_of_metrics)}
				all_useful_jobs_infos.append(useful_job_info_2)
			return(all_useful_jobs_infos)

class CanaryTrainingRegressionAnalysis:
		def __init__(self):
			'''takes in an already instansiated canary kick off_object'''
			#self.__dict__=copy.deepcopy(canary_kickoff_object.__dict__)
			pass
		def turn_info_to_dataframe(self,useful_jobs_info=None):
			df_1=pandas.DataFrame.from_records(useful_jobs_info)
			return(df_1)
		def get_bootstraps(self,x=None,y=None,num_bootstraps=None):
			#NEED TO ADD FUNCTIONALITY FOR STANDARD DEVIATION
			list_of_x_y=list(zip(x,y))
			#logger.error('The original data')
			#logger.error(list_of_x_y)
			list_size=len(list_of_x_y)
			all_bootstraps=[]
			#in order to avoid statistical issues, cap list size
			if list_size>10:
				list_size=10
			for i in range(0,num_bootstraps):
				a_bootstrap=choices(list_of_x_y,k=list_size)
				the_dataframe=pandas.DataFrame.from_records(a_bootstrap)
				the_dataframe_2=the_dataframe.groupby(0).mean()
				is_monotonic=the_dataframe_2[1].is_monotonic
				if is_monotonic==True:
					all_bootstraps.append(a_bootstrap)
			return(all_bootstraps)
		def perform_extrapolation(self,the_dataframe=None,col_to_extrapolate=None,instance_type=None,force_increase=False):
			'''force_increase forces an increase in the predicted extrapolation value; no decreasing or same value'''
			extrapolation_based_on="PercentageDataTrainedOn"
			mini_data_frame=the_dataframe[the_dataframe['InstanceType']==instance_type]
			x=mini_data_frame[extrapolation_based_on].tolist()
			x=[float(i) for i in x] #covert to float, since it was string before
			x=numpy.asarray(x)
			y=mini_data_frame[col_to_extrapolate].tolist()
			y=numpy.asarray(y)
			#print(y)

			#to force an increase in the extrapolation take mean as true value and create slope
			if force_increase==True:
				#use a percentile of y to be more conservative on training times
				y_mean=numpy.percentile(y,75) #not actually the mean
				x_mean=numpy.mean(x)
				normalized_x=[i/x_mean for i in x]
				#print("normalized_x")
				#print(normalized_x)
				pseudo_y=[i*y_mean for i in normalized_x]
				y=pseudo_y
				#y=numpy.asarray(y)
				#print("adjusted y")
				#print(y)
				#print("new")
			#create the model

			y_modification_parameter=1.1 #parameter for how much to inflate y values before regression.
			y=y*y_modification_parameter
			the_bootstraps=self.get_bootstraps(x=x,y=y,num_bootstraps=1000)
			all_results=[]
			#logger.error('The first bootstrap')
			#logger.error(the_bootstraps[0])
			all_zero_results=[] #for debugging
			for i in the_bootstraps:
				x_and_y=list(zip(*i))
				x=numpy.asarray(x_and_y[0])
				y=numpy.asarray(x_and_y[1])
				reg = LinearRegression().fit(x.reshape(-1, 1), y)
				#perform extrapolation on all data
				all_data_percentage=1
				the_result=reg.predict(numpy.asarray([all_data_percentage]).reshape(1, -1) )
				all_results.append(the_result[0])

				no_data_percentage=0 #for debugging
				the_result=reg.predict(numpy.asarray([no_data_percentage]).reshape(1, -1) )
				all_zero_results.append(the_result[0])

			#logger.error('All bootstrap results')
			#logger.error(all_results)
			#logger.error('All zero results')
			#logger.error(all_zero_results)
			final_result=numpy.mean(all_results) #get mean of all the bootstraps
			return (final_result)

		#def jitter_to_be_monotonic(self,x=None)
		def perform_all_extrapolations(self,the_dataframe=None):
			cols_to_extrapolate=['CPUUtilization',"MemoryUsedPercent","TrainingTimeInSeconds",'GPUUtilization','GPUMemoryUtilization'] #the columns we want to extrapolate
			cols_to_force_increase=["TrainingTimeInSeconds"]
			the_instance_types=list(set(the_dataframe['InstanceType'].tolist()))
			dict_of_results={}
			for instance_type in the_instance_types:
				#dict_of_results[instance_type]=[{}]
				dict_of_results[instance_type]={}
				for col_to_extrapolate in cols_to_extrapolate:
					#print(col_to_extrapolate)
					if col_to_extrapolate in cols_to_force_increase:
						the_result=self.perform_extrapolation(the_dataframe=the_dataframe,
							col_to_extrapolate=col_to_extrapolate,instance_type=instance_type, force_increase=False)
					else:
						the_result=self.perform_extrapolation(the_dataframe=the_dataframe,
							col_to_extrapolate=col_to_extrapolate,instance_type=instance_type, force_increase=False)
					projected_colname=f"Projected_{col_to_extrapolate}"
					#dict_of_results[instance_type][0][projected_colname]=the_result
					dict_of_results[instance_type][projected_colname]=the_result
			return(dict_of_results)
		def correct_for_time_to_download_image(self,the_dataframe=None):
			'''subtract the amount it takes to download the image, which is mostly contant'''
			amount_of_time_to_download_image=50  #assume set amount of time
			the_dataframe['TrainingTimeInSeconds']=the_dataframe['TrainingTimeInSeconds']-amount_of_time_to_download_image
			return(the_dataframe)
		def convert_to_dataframe(self,the_dict=None):
			df_1=pandas.read_json(json.dumps(the_dict),orient="index")
			return(df_1)
		def add_instance_price_info_and_projected_total_cost(self,df_projected_resource_use=None):
			the_row_names=list(df_projected_resource_use.index)
			df_projected_resource_use['price']=None
			df_projected_resource_use['Projected_TotalCost']=None
			for i in the_row_names:
				the_price=self.get_instance_pricing(instance_type=i)
				df_projected_resource_use.at[i, 'price'] =the_price
			df_projected_resource_use['Projected_TotalCost']=df_projected_resource_use['Projected_TrainingTimeInSeconds']*df_projected_resource_use['price']
			return(df_projected_resource_use)
		def select_instance_from_projected_use(self,df_projected_resource_use=None,mem_upper_thresh=80,
			cpu_upper_thresh=120,gpu_upper_thresh=120,gpu_mem_upper_thresh=80,mem_lower_thresh=-1000,
			cpu_lower_thresh=-1000,gpu_lower_thresh=-1000,gpu_mem_lower_threshold=-1000):
			'''
			Given projected metrics, auto-select an instance to use. Filters out instances with projected over and under use, 
			then ranks the remaining by projected training cost and selects the cheapest one
			'''
			try:
				#df_next=df_projected_resource_use[(df_projected_resource_use['Projected_MemoryUsedPercent'] <mem_upper_thresh) & (df_projected_resource_use['Projected_MemoryUsedPercent'] >=mem_lower_thresh) ]
				#df_next=df_projected_resource_use[(df_projected_resource_use['Projected_CPUUtilization'] <cpu_upper_thresh) & (df_projected_resource_use['Projected_CPUUtilization'] >=cpu_lower_thresh) ]
				#df_next=df_projected_resource_use[(df_projected_resource_use['Projected_GPUUtilization'] <gpu_upper_thresh) & (df_projected_resource_use['Projected_GPUUtilization'] >=gpu_lower_thresh) ]
				#df_next=df_projected_resource_use[(df_projected_resource_use['Projected_GPUMemoryUtilization'] <gpu_mem_upper_thresh) & (df_projected_resource_use['Projected_GPUMemoryUtilization'] >=gpu_lower_thresh) ]

				#create a score for the instances
				#num=df_projected_resource_use._get_numeric_data()
				#num[num <= 0] = 1
				#df_projected_resource_use

				#add a multiplier so that saving extra time is rewarded in an exponential manner.
				multiplier=2
				df_next=df_projected_resource_use
				to_sort_by=df_next['Projected_TrainingTimeInSeconds']**multiplier*df_next['price']
				best_instance_selected=df_next.loc()[to_sort_by.sort_values().index].index[0]
				#best_instance_selected=df_next.sort_values(by='Projected_TotalCost').iloc()[0].name


			except Exception as e:
				print(e)
				best_instance_selected=None
				print("Error. Possibly no instance found that satisfies criteria.")
			return(best_instance_selected)
		def get_instance_pricing(self, instance_type=None):
			'''Get SageMaker instance price'''
			input_file="instance_price_info.csv"
			package_directory = os.path.dirname(__file__)
			df_1=pandas.read_csv(os.path.join(package_directory,input_file))
			#df_1=pandas.read_csv(input_file)
			instance_price=df_1[df_1['instance_type']==instance_type]['price']
			instance_price=float(instance_price)/(60*60) #get price per second
			return(instance_price)

class CanaryTraining():
	'''
	data_location (str): the s3 location of the input training data
	output_s3_location (str): the s3 location to put the output canary training statistics. Exclude the final "/"
	temp_dir (str): the local temporary directory where manifest data is stored. This will be copied as a subdirectory of  {output_s3_location}
	Note that if you already have canary training experiments done, or manifest files created, you can point to an already existing directory.
	instance_types (list): a list of all instance types to test
	training_percentages (list): a list of percentages of data to sample
	estimator(sagemaker.estimator.Estimator): A constructed SageMaker estimator

	Example:
	data_location='s3://aws-hcls-ml/public_assets_support_materials/canary_training_data/taxi_yellow_trip_data_processed'
	output_s3_location=f"s3://an_example_bucket/taxi_output_data/"
	temp_dir="temp_dir_123"
	instance_types=["ml.m5.24xlarge"]
	training_percentages=[.01,.01,.01,.02,.02,.02,.03,.03,.03]

	ct=CanaryTraining(data_location=data_location,output_s3_location=output_s3_location,
                           the_temp_dir=the_temp_dir,instance_types=instance_types,estimator=estimator,training_percentages=training_percentages)
	'''
	def __init__(self,data_location=None,output_s3_location=None,the_temp_dir=None,instance_types=None,training_percentages=None,estimator=None):
		self.data_location=data_location
		self.output_s3_location=output_s3_location
		self.the_temp_dir=the_temp_dir
		self.instance_types=instance_types
		self.training_percentages=training_percentages
		self.estimator=estimator

	def prepare_canary_training_data(self):
		'''prepare the underlying data for canary training, given the data location and the
		sagemaker estimator'''
		ctk=CanaryTrainingKickoff(data_location=self.data_location,
                          output_s3_location=self.output_s3_location,
                          the_temp_dir=self.the_temp_dir,
                          instance_types=self.instance_types,
                          training_percentages=self.training_percentages)
		the_uris=ctk.list_data_location(s3_uri=ctk.data_location)
		sampled_uris_list_list=ctk.sample_uris(the_uris=the_uris)
		ctk.create_manifests_for_uris(the_uris_list_list=sampled_uris_list_list)
		ctk.copy_directory_to_s3(the_directory=ctk.temp_dir)

		self.estimators_list=ctk.create_estimators(base_estimator=self.estimator)
		#manifest_training_inputs=ctk.create_manifest_training_inputs(content_type='csv')
		manifest_dir=f'{ctk.output_s3_location}/{ctk.temp_dir}/manifests'
		manifest_locations=ctk.list_data_location(s3_uri=manifest_dir)
		self.manifest_training_inputs_list=[sagemaker.inputs.TrainingInput(i,s3_data_type='ManifestFile', content_type='csv' ) for i in manifest_locations]
		self.ctk=ctk

	def kick_off_canary_training_jobs(self,wait=False,training_channels_list=['train','test','validation']):
		'''create the canary SageMaker Training jobs.
		wait (default - False): Default of False runs all jobs in parallel. Set "wait"
		equal to true to run the jobs serially
		training_channels_list: The training channels to use for the SageMaker estimator. By default, all three will must be specified. 
		For example, if you have only a training channel, then do training_channels_list=['train']
		'''
		submitted_jobs_information=self.ctk.kick_off_canary_training(estimators_list=self.estimators_list,
                                                        manifest_training_inputs_list=self.manifest_training_inputs_list,
                                                        training_channels_list=training_channels_list,
                                                        estimator_kwarks={"wait":wait} #uncomment to run serially                                                     
                                                       )
		self.ctk.copy_canary_training_job_info_to_s3(all_jobs_l=submitted_jobs_information)

	def get_predicted_resource_consumption(self):
		'''get the resource predictions for the entire training jobs, and return the raw data as well'''
		#submitted_jobs_information
		submitted_jobs_information=self.ctk.get_canary_training_jobs_list_from_temp_dir()
		cjp=GetJobProfileInfo()
		useful_jobs_info=cjp.get_useful_jobs_info(submitted_jobs_information=submitted_jobs_information)
		 # Now perform regression analysis to extrapolate resource use
		ctra=CanaryTrainingRegressionAnalysis()
		the_dataframe=ctra.turn_info_to_dataframe(useful_jobs_info)
		dict_results=ctra.perform_all_extrapolations(the_dataframe=the_dataframe)
		df_projected_resource_use=ctra.convert_to_dataframe(dict_results)
		df_projected_resource_use=ctra.add_instance_price_info_and_projected_total_cost(df_projected_resource_use=df_projected_resource_use)
		#instance_selected=ctra.select_instance_from_projected_use(df_projected_resource_use=df_projected_resource_use)
		raw_canary_job_info_df=the_dataframe
		df_projected_resource_use.head()
		return(df_projected_resource_use,raw_canary_job_info_df)

