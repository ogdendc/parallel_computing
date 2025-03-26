# Databricks notebook source
# MAGIC %md
# MAGIC # The groupBy.applyInPandas method 
# MAGIC > ## applies an instance of a Pandas UDF separately to each groupBy subset of a Spark DataFrame; 
# MAGIC > ## allowing pandas-based engineering and modeling to run in parallel.
# MAGIC > ## This notebook also demonstrates some functionality of creating nested MLflow experiments,
# MAGIC > ## and controlling what gets logged to mlflow.

# COMMAND ----------

# DBTITLE 1,reading a csv into a spark dataframe
# reading the sample data into a spark dataframe, and dropping the index column
spark_df_diamonds = spark.read.csv("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header="true", inferSchema="true").drop("_c0")
display(spark_df_diamonds)

# COMMAND ----------

# MAGIC %md ## going to group by "cut", but first let's combine the "Good" and "Fair" diamonds into one group...
# MAGIC > ## for a somewhat more balanced split across groups

# COMMAND ----------

from pyspark.sql.functions import when

spark_df_diamonds = spark_df_diamonds.withColumn("cut", when((spark_df_diamonds.cut == "Good") | (spark_df_diamonds.cut == "Fair"),"Good_or_Fair").otherwise(spark_df_diamonds.cut))

# COMMAND ----------

spark_df_diamonds.groupBy("cut").count().show()

# COMMAND ----------

# We want to fit each model in parallel using separate Spark tasks. 
# When working with smaller groups of data, Adaptive Query Execution (AQE) can combine these smaller model fitting tasks into a single, larger task where models are fit sequentially. Since we want to avoid this behavior in this example, we will disable Adaptive Query Execution. Generally, AQE should be left enabled.

spark.conf.set('spark.sql.adaptive.enabled', 'false')

# COMMAND ----------

# Also, since we are using Python libraries that can benefit from multiple cores, we can instruct Spark to provide more than one CPU core per tasks by setting **spark.task.cpus** in the Advanced options of the Clusters UI. In the Spark config section under the Spark tab, we set **spark.task.cpus 4**. In our example, we will fit 4 models in parallel, so we need 16 cores in total to fit all models at the same time. 
# We could choose compute optimized instances if our groupBy operations (in our udf) require computational intensity.

# COMMAND ----------

# creating x and y sets for model training...and converting from spark to pandas 
x_pd_df_diamonds = spark_df_diamonds.drop("price").toPandas()
y_pd_df_diamonds = spark_df_diamonds.select("price").toPandas()

# COMMAND ----------

from sklearn.model_selection import train_test_split

x_pd_train, x_pd_test, y_pd_train, y_pd_test = train_test_split(x_pd_df_diamonds ,y_pd_df_diamonds, test_size=0.3, random_state=42)

# COMMAND ----------

display(x_pd_train)

# COMMAND ----------

# resetting index so that joining below, with onehot_df, will work
x_pd_train = x_pd_train.reset_index(drop=True)
x_pd_test = x_pd_test.reset_index(drop=True)

# COMMAND ----------

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
                        
for column in ['cut', 
               'color',
               'clarity']:
    
    x_pd_train[column] = x_pd_train[column].fillna("X")   # replacing any missing values, just in case
    
    x_pd_train[column] = x_pd_train[column].str.replace(' ', '')   # stripping all blanks from values, just in case
    
    # training the encoder using the train set
    result = encoder.fit(x_pd_train[column].to_numpy().reshape(-1, 1))
    
    # getting post-encoded column names from the encoder
    column_name = encoder.get_feature_names_out([column])
                                                 
    # one-hot-encoding the training data
    onehot = encoder.transform(x_pd_train[column].to_numpy().reshape(-1, 1))
    onehot_df = pd.DataFrame(onehot, columns=column_name)
    x_pd_train = x_pd_train.join(onehot_df)
    
    # one-hot-encoding the test data
    onehot = encoder.transform(x_pd_test[column].to_numpy().reshape(-1, 1))
    onehot_df = pd.DataFrame(onehot, columns=column_name)
    x_pd_test = x_pd_test.join(onehot_df)
    
    #filename = r'/vcd/warehouse/default/pickle/FVDT/'+column+'encoder.sav' # play around with saving to DBFS
    #pickle.dump(result,open(filename,'wb'))
    
print("columns in new input data ... check to make sure ok")
print(" ") 
print(x_pd_train.dtypes)

# COMMAND ----------

import numpy as np 

# dropping categoricals
x_pd_train = x_pd_train.drop(['cut', 'color', 'clarity'], axis=1)
x_pd_test  = x_pd_test.drop(['cut', 'color', 'clarity'], axis=1)

# converting y to array simply to avoid warning in the log
y_pd_train_array = np.ravel(y_pd_train)
y_pd_test_array = np.ravel(y_pd_test)

# COMMAND ----------

x_pd_train.isnull().sum()

# COMMAND ----------

# MAGIC %md ## Training an overall (no groupBy yet) gradient boosting regressor model

# COMMAND ----------

# DBTITLE 1,Turning on MLflow auto-logging
import mlflow

mlflow.autolog()

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# n_estimators = number of iterations...should be less than 100?
# min_samples_split should be ~0.5-1% of total values. 
# min_samples_leaf can be selected based on intuition. 
# max_depth = 8 should be chosen (5-8) based on the number of observations and predictors. 
# max_features = ‘sqrt’ : Its a general thumb-rule to start with square root.
# subsample = 0.8 : This is a commonly used used start value.
# validation_fraction, default=0.1. Proportion of training data to set aside as validation set for early stopping. Only used if n_iter_no_change is set to an integer.
# n_iter_no_changeint, default=None
#    n_iter_no_change is used to decide if early stopping will be used to terminate training when validation score is not improving. By default it is set to None to disable early stopping. #    If set to a number, it will set aside validation_fraction size of the training data as validation and terminate training when validation score is not improving in all of the previous #    n_iter_no_change numbers of iterations. Values must be in the range [1, inf).
# tol, default=1e-4. Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops. Values must #    be in the range (0.0, inf).
# loss{‘squared_error’, ‘absolute_error’, ‘huber’, ‘quantile’}, default=’squared_error’
#    Loss function to be optimized. ‘squared_error’ refers to the squared error for regression. ‘absolute_error’ refers to the absolute error of regression and is a robust loss function. 
#    ‘huber’ is a combination of the two. ‘quantile’ allows quantile regression (use alpha to specify the quantile).
    
params = {'n_estimators': 300, 'learning_rate':0.1, 'min_samples_split':100, 'min_samples_leaf':25, 'max_depth':8, 'max_features':'sqrt','subsample':0.8,'random_state':10,
         'n_iter_no_change': 5, 'validation_fraction': 0.2, 'tol': 0.001}

GBR_model = GradientBoostingRegressor(**params)

GBR_model.fit(x_pd_train, y_pd_train_array)

# scoring the training and test sets and calculating goodness-of-fit metrics
y_train_pred = GBR_model.predict(x_pd_train)
rmse_train = mean_squared_error(y_pd_train, y_train_pred, squared=False)

y_test_pred = GBR_model.predict(x_pd_test)
rmse_test = mean_squared_error(y_pd_test, y_test_pred, squared=False)
 
print("Number of boosting iterations before early stopping = ", GBR_model.n_estimators_)

print("RMSE train: %.2f" % rmse_train)
print("RMSE test: %.2f" % rmse_test)

# COMMAND ----------

import matplotlib.pyplot as plt

train_RMSE = np.zeros((params['n_estimators'],), dtype=np.float64)
test_RMSE = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(GBR_model.staged_predict(x_pd_train)):
    train_RMSE[i] = mean_squared_error(y_pd_train_array, y_pred, squared=False)

for i, y_pred in enumerate(GBR_model.staged_predict(x_pd_test)):
    test_RMSE[i] = mean_squared_error(y_pd_test_array, y_pred, squared=False)

plt.figure(figsize=(30,15))
plt.subplot(1, 2, 1)
plt.title('RMSE by iteration')
plt.plot(np.arange(params['n_estimators']) + 1, train_RMSE, 'b-',
         label='Training Set RMSE')
plt.plot(np.arange(params['n_estimators']) + 1, test_RMSE, 'r-',
         label='Test Set RMSE')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('RMSE')

# COMMAND ----------

# MAGIC %md ### Take a look at the experiment logged to mlflow...
# MAGIC > ### with funky default name, and our 'MSE by iteration' plot not being logged by default.

# COMMAND ----------

# MAGIC %md ### So, below, we are going to redo the same model...
# MAGIC > ### while controlling what gets logged to mlflow.

# COMMAND ----------

from sklearn.metrics import mean_absolute_error

params = {'n_estimators': 300, 'learning_rate':0.1, 'min_samples_split':100, 'min_samples_leaf':25, 'max_depth':8, 'max_features':'sqrt','subsample':0.8,'random_state':10,
         'n_iter_no_change': 5, 'validation_fraction': 0.2, 'tol': 0.001}

GBR_model = GradientBoostingRegressor(**params)


with mlflow.start_run(run_name="Overall GBR Model"):
  mlflow.autolog()
  
  GBR_model.fit(x_pd_train, y_pd_train_array)

  # scoring the training and test sets and calculating goodness-of-fit metrics
  y_train_pred = GBR_model.predict(x_pd_train)
  rmse_train = mean_squared_error(y_pd_train, y_train_pred, squared=False)
  abs_err_train = mean_absolute_error(y_pd_train, y_train_pred)

  y_test_pred = GBR_model.predict(x_pd_test)
  rmse_test = mean_squared_error(y_pd_test, y_test_pred, squared=False)
  abs_err_test = mean_absolute_error(y_pd_test, y_test_pred)
 
  print("Number of boosting iterations before early stopping = ", GBR_model.n_estimators_)

  print("RMSE train: %.2f" % rmse_train)
  print("RMSE test: %.2f" % rmse_test)
  
  # creating a plot to log
  train_RMSE = np.zeros((params['n_estimators'],), dtype=np.float64)
  test_RMSE = np.zeros((params['n_estimators'],), dtype=np.float64)

  for i, y_pred in enumerate(GBR_model.staged_predict(x_pd_train)):
      train_RMSE[i] = mean_squared_error(y_pd_train_array, y_pred, squared=False)

  for i, y_pred in enumerate(GBR_model.staged_predict(x_pd_test)):
      test_RMSE[i] = mean_squared_error(y_pd_test_array, y_pred, squared=False)

  plt.figure(figsize=(30,15))
  plt.subplot(1, 2, 1)
  plt.title('RMSE by iteration')
  plt.plot(np.arange(params['n_estimators']) + 1, train_RMSE, 'b-',
           label='Training Set RMSE')
  plt.plot(np.arange(params['n_estimators']) + 1, test_RMSE, 'r-',
           label='Test Set RMSE')
  plt.legend(loc='upper right')
  plt.xlabel('Boosting Iterations')
  plt.ylabel('RMSE')
  
  plt.savefig("RMSE_iteration_plot.png", bbox_inches="tight")
  plt.close()
  
  mlflow.log_artifact("RMSE_iteration_plot.png")
  mlflow.set_tag("model", "overall GBR model")
  
  # Notice that with autologging, these metrics are already be logged by default...
  #    so this code is just to demonstrate the ability to control what gets logged and how it's named:
  mlflow.log_metric('Mean Absolute Error TRAIN', abs_err_train)  # calculated above
  mlflow.log_metric('Mean Absolute Error TEST', abs_err_test)
  

# COMMAND ----------

# MAGIC %md ## Now to do groupBy training
# MAGIC > ## ...initially via *serial* iteration, just for comparative purposes.

# COMMAND ----------

groups = ['Premium', 'Ideal', 'Very Good', 'Good_or_Fair']  # group by diamond 'cut'

for g in groups:
  
  spark_df_group = spark_df_diamonds.filter(spark_df_diamonds.cut == g)
  
  # creating x and y sets for model training...and converting from spark to Pandas 
  x_pd_df_diamonds = spark_df_group.drop("price").toPandas()
  y_pd_df_diamonds = spark_df_group.select("price").toPandas()
  
  # partitioning into train and test sets
  x_pd_train, x_pd_test, y_pd_train, y_pd_test = train_test_split(x_pd_df_diamonds ,y_pd_df_diamonds, test_size=0.3, random_state=42)
  
  # resetting index so that joining below, with onehot_df, will work
  x_pd_train = x_pd_train.reset_index(drop=True)
  x_pd_test = x_pd_test.reset_index(drop=True)
  
  # one-hot encoding:
  for column in ['color',
               'clarity']:
    
    x_pd_train[column] = x_pd_train[column].fillna("X")   # replacing any missing values, just in case
    
    x_pd_train[column] = x_pd_train[column].str.replace(' ', '')   # stripping all blanks from values, just in case
    
    # training the encoder using the train set
    result = encoder.fit(x_pd_train[column].to_numpy().reshape(-1, 1))
    
    # getting post-encoded column names from the encoder
    column_name = encoder.get_feature_names_out([column])

    # one-hot-encoding the training data
    onehot = encoder.transform(x_pd_train[column].to_numpy().reshape(-1, 1))
    onehot_df = pd.DataFrame(onehot, columns=column_name)
    x_pd_train = x_pd_train.join(onehot_df)
    
    # one-hot-encoding the test data
    onehot = encoder.transform(x_pd_test[column].to_numpy().reshape(-1, 1))
    onehot_df = pd.DataFrame(onehot, columns=column_name)
    x_pd_test = x_pd_test.join(onehot_df)
    
  # dropping categoricals
  x_pd_train = x_pd_train.drop(['cut', 'color', 'clarity'], axis=1)
  x_pd_test  = x_pd_test.drop(['cut', 'color', 'clarity'], axis=1)

  # converting y to array simply to avoid warning in the log
  y_pd_train_array = np.ravel(y_pd_train)
  y_pd_test_array = np.ravel(y_pd_test)
    
  # setting hyperparameters, fitting model, and evaluating goodness-of-fit for each group
  params = {'n_estimators': 300, 'learning_rate':0.1, 'min_samples_split':100, 'min_samples_leaf':25, 'max_depth':8, 'max_features':'sqrt','subsample':0.8,'random_state':10,
            'n_iter_no_change': 5, 'validation_fraction': 0.2, 'tol': 0.001}

  GBR_model = GradientBoostingRegressor(**params)

  GBR_model.fit(x_pd_train, y_pd_train_array)

  # scoring the training and test sets and calculating goodness-of-fit metrics
  y_train_pred = GBR_model.predict(x_pd_train)
  rmse_train = mean_squared_error(y_pd_train, y_train_pred, squared=False)

  y_test_pred = GBR_model.predict(x_pd_test)
  rmse_test = mean_squared_error(y_pd_test, y_test_pred, squared=False)
 
  print(" ")
  print("   Group = ", g)
  print("   Number of boosting iterations before early stopping = ", GBR_model.n_estimators_)
  
  print("   RMSE train: ", '{:,.2f}'.format(rmse_train))
  print("   RMSE test: ",  '{:,.2f}'.format(rmse_test))

# COMMAND ----------

# MAGIC %md ### Again, take note that these models are auto-logged with funky names.
# MAGIC > ### plus, there's no indication that they are "child" models that should be grouped/organized together.

# COMMAND ----------

# MAGIC %md ### So let's modify to better control how these models show up in MLflow tracking...

# COMMAND ----------

# MAGIC %md #### Notice in this *serial* version of the modeling, we are nesting each model (by group)
# MAGIC > #### under a parent model...just to keep things more organized in MLflow tracking output.

# COMMAND ----------

groups = ['Premium', 'Ideal', 'Very Good', 'Good_or_Fair']  # group by diamond 'cut'

with mlflow.start_run(run_name="Serial Iteration Parent") as parent_run:
  mlflow.autolog()
  
  for g in groups:
    with mlflow.start_run(run_name="Serial GBR Group = "+str(g), nested=True):
      
      mlflow.log_param('by_group', g)  
  
      spark_df_group = spark_df_diamonds.filter(spark_df_diamonds.cut == g)
  
      # creating x and y sets for model training...and converting from spark to pandas 
      x_pd_df_diamonds = spark_df_group.drop("price").toPandas()
      y_pd_df_diamonds = spark_df_group.select("price").toPandas()
  
      # partitioning into train and test sets
      x_pd_train, x_pd_test, y_pd_train, y_pd_test = train_test_split(x_pd_df_diamonds ,y_pd_df_diamonds, test_size=0.3, random_state=42)
  
      # resetting index so that joining below, with onehot_df, will work
      x_pd_train = x_pd_train.reset_index(drop=True)
      x_pd_test = x_pd_test.reset_index(drop=True)
  
      # one-hot encoding:
      for column in ['color', 'clarity']:
        
        x_pd_train[column] = x_pd_train[column].fillna("X")   # replacing any missing values, just in case
    
        x_pd_train[column] = x_pd_train[column].str.replace(' ', '')   # stripping all blanks from values, just in case
    
        # training the encoder using the train set
        result = encoder.fit(x_pd_train[column].to_numpy().reshape(-1, 1))
    
        # getting post-encoded column names from the encoder
        column_name = encoder.get_feature_names_out([column])
    
        # one-hot-encoding the training data
        onehot = encoder.transform(x_pd_train[column].to_numpy().reshape(-1, 1))
        onehot_df = pd.DataFrame(onehot, columns=column_name)
        x_pd_train = x_pd_train.join(onehot_df)
    
        # one-hot-encoding the test data
        onehot = encoder.transform(x_pd_test[column].to_numpy().reshape(-1, 1))
        onehot_df = pd.DataFrame(onehot, columns=column_name)
        x_pd_test = x_pd_test.join(onehot_df)
    
      # dropping categoricals
      x_pd_train = x_pd_train.drop(['cut', 'color', 'clarity'], axis=1)
      x_pd_test  = x_pd_test.drop(['cut', 'color', 'clarity'], axis=1)

      # converting y to array simply to avoid warning in the log
      y_pd_train_array = np.ravel(y_pd_train)
      y_pd_test_array = np.ravel(y_pd_test)
    
      # setting hyperparameters, fitting model, and evaluating goodness-of-fit for each group
      params = {'n_estimators': 300, 'learning_rate':0.1, 'min_samples_split':100, 'min_samples_leaf':25, 'max_depth':8, 'max_features':'sqrt','subsample':0.8,'random_state':10,
                'n_iter_no_change': 5, 'validation_fraction': 0.2, 'tol': 0.001}

      GBR_model = GradientBoostingRegressor(**params)

      GBR_model.fit(x_pd_train, y_pd_train_array)

      # scoring the training and test sets and calculating goodness-of-fit metrics
      y_train_pred = GBR_model.predict(x_pd_train)
      rmse_train = mean_squared_error(y_pd_train, y_train_pred, squared=False)

      y_test_pred = GBR_model.predict(x_pd_test)
      rmse_test = mean_squared_error(y_pd_test, y_test_pred, squared=False)
 
      print(" ")
      print("   Group = ", g)
      print("   Number of boosting iterations before early stopping = ", GBR_model.n_estimators_)
  
      print("   RMSE train: ", '{:,.2f}'.format(rmse_train))
      print("   RMSE test: ",  '{:,.2f}'.format(rmse_test))
      
      mlflow.set_tag("model", "serial GBR group = "+str(g))


# COMMAND ----------

# MAGIC %md ## Using the above as the basis to form our UDF for the group-by distributed modeling.

# COMMAND ----------

# MAGIC %md #### Notice in this *parallel* version of the modeling, below, we are nesting each model (by group)
# MAGIC > #### under a parent model...however...
# MAGIC > #### unlike above, we are explicitly specifying in the function to which parent we assign
# MAGIC > #### each group/model...since the nesting is not inherently inferred with this groupby approach.
# MAGIC > #### Check out this tutorial for more info, including how to incorporate hyperparameter optimization within each groupby:
# MAGIC >> #### https://github.com/marshackVB/parallel_models_blog

# COMMAND ----------

from collections import OrderedDict
import datetime

"""
  Configure a PandasUDF function and that trains a model on a group of data. 
  The UDF is applied using the groupBy.applyInPandas method.
"""
  
def train_model_udf(group_training_data):
  
  # Measure the training time of each model
  start = datetime.datetime.now()
   
  # Capture the name of the group to be modeled
  g = group_training_data['cut'].loc[0]
    
  with mlflow.start_run(run_name="Parallel GBR Group = "+str(g), nested=True):
    mlflow.autolog()
    
    mlflow.log_param('by_group', g)  
  
    #spark_df_group = spark_df_diamonds.filter(spark_df_diamonds.cut == g)  # inherent in the groupBy
  
    # creating x and y sets for model training...
    #    where the data is already now in pandas dataframe
    x_pd_df_diamonds = group_training_data.drop(['price'], axis=1)
    y_pd_df_diamonds = group_training_data[['price']]
  
    # partitioning into train and test sets
    x_pd_train, x_pd_test, y_pd_train, y_pd_test = train_test_split(x_pd_df_diamonds, y_pd_df_diamonds, test_size=0.3, random_state=42)
  
    # resetting index so that joining below, with onehot_df, will work
    x_pd_train = x_pd_train.reset_index(drop=True)
    x_pd_test = x_pd_test.reset_index(drop=True)
  
    # one-hot encoding:
    for column in ['color',
                 'clarity']:
    
      x_pd_train[column] = x_pd_train[column].fillna("X")   # replacing any missing values, just in case
    
      x_pd_train[column] = x_pd_train[column].str.replace(' ', '')   # stripping all blanks from values, just in case
    
      # training the encoder using the train set
      result = encoder.fit(x_pd_train[column].to_numpy().reshape(-1, 1))
    
      # getting post-encoded column names from the encoder
      column_name = encoder.get_feature_names_out([column])
    
      # one-hot-encoding the training data
      onehot = encoder.transform(x_pd_train[column].to_numpy().reshape(-1, 1))
      onehot_df = pd.DataFrame(onehot, columns=column_name)
      x_pd_train = x_pd_train.join(onehot_df)
    
      # one-hot-encoding the test data
      onehot = encoder.transform(x_pd_test[column].to_numpy().reshape(-1, 1))
      onehot_df = pd.DataFrame(onehot, columns=column_name)
      x_pd_test = x_pd_test.join(onehot_df)
    
    # dropping categoricals
    x_pd_train = x_pd_train.drop(['cut', 'color', 'clarity'], axis=1)
    x_pd_test  = x_pd_test.drop(['cut', 'color', 'clarity'], axis=1)

    # converting y to array simply to avoid warning in the log
    y_pd_train_array = np.ravel(y_pd_train)
    y_pd_test_array = np.ravel(y_pd_test)
    
    # setting hyperparameters, fitting model, and evaluating goodness-of-fit for each group
    params = {'n_estimators': 300, 'learning_rate':0.1, 'min_samples_split':100, 'min_samples_leaf':25, 'max_depth':8,
              'max_features':'sqrt','subsample':0.8,'random_state':10,
              'n_iter_no_change': 5, 'validation_fraction': 0.2, 'tol': 0.001}

    GBR_model = GradientBoostingRegressor(**params)

    GBR_model.fit(x_pd_train, y_pd_train_array)

    # scoring the training and test sets and calculating goodness-of-fit metrics
    y_train_pred = GBR_model.predict(x_pd_train)
    rmse_train = mean_squared_error(y_pd_train, y_train_pred, squared=False)

    y_test_pred = GBR_model.predict(x_pd_test)
    rmse_test = mean_squared_error(y_pd_test, y_test_pred, squared=False)
    
    mlflow.log_metric('Root Mean Squared Error TEST', rmse_test)

    end = datetime.datetime.now()
    elapsed = end-start
    minutes = round(elapsed.total_seconds() / 60, 2)
    
    # Capture data about our the model
    digits = 1
    metrics = OrderedDict()
    metrics["rmse_train"]=       round(mean_squared_error(y_pd_train, y_train_pred, squared=False), digits)
    metrics["rmse_test"]=        round(mean_squared_error(y_pd_test, y_test_pred, squared=False), digits)
    metrics["boosting_iterations"]= GBR_model.n_estimators_
    
    other_meta = OrderedDict()
    other_meta['group'] =           g
    other_meta['start_time'] =      start.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    #other_meta['end_time'] =        end.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    other_meta['elapsed_minutes'] = minutes
    
    other_meta.update(metrics)
    
    # this line is key to get these disributed child runs associated with parent
    mlflow.set_tag("mlflow.parentRunId", parent_run2.info.run_id)
      
    return pd.DataFrame(other_meta, index=[0])
  
  #return train_model_udf

# COMMAND ----------

# The PandasUDF returns a Pandas DataFrame; we must specify a Spark DataFrame schema that maps to the column names and Python data types returned by the UDF. 

from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, ArrayType, MapType

# Specify Spark DataFrame Schema
spark_types = [('group',                 StringType()),
               ('start_time',            StringType()),
               #('end_time',              StringType()),
               ('elapsed_minutes',       FloatType()),
               ('boosting_iterations',   IntegerType()),
               ('rmse_train',            FloatType()),
               ('rmse_test',             FloatType())
              ]

spark_schema = StructType()

for col_name, spark_type in spark_types:
  spark_schema.add(col_name, spark_type)

# COMMAND ----------

# here we are defining the parent run
with mlflow.start_run(run_name="Parallel GBR Parent") as parent_run2:
  mlflow.autolog()

# applying the PandasUDF
best_stats = spark_df_diamonds.groupBy('cut').applyInPandas(train_model_udf, schema=spark_schema)

# thanks to spark's lazy evaluation...
#    this command will actually execute the udf...the *parallel* model training:
display(best_stats)

# COMMAND ----------

# MAGIC %md #Compare the *parallel* runtime here with the *serial* runtime further above.
# MAGIC > ## Note: for small datasets, there may not be a big difference...due to overhead.

# COMMAND ----------

# NEXT STEPS:  
#   enhance to incorporate hyperparameter optimizations
