# Credit Card Default Prediction App

## OBJECTIVE
The objective of this project is to create and deploy an application for my classification model. During the model training process, I selected the best classfification model and incorporated it into a pipeline to better predict the risk of credit card payment defaults. 


## Dataset
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. 

For more information on the dataset, please visit the UCI ML Repository
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

or Kaggle website at https://www.kaggle.com/code/selener/prediction-of-credit-card-default/input

## Machine Learning Pipeline
1. Analyze Data: In this initial step, we attempted to comprehend the data and searched for various available features. We looked for things like the shape of the data, the data types of each feature, a statistical summary, etc. at this stage.
2. EDA: EDA stands for Exploratory Data Analysis. It is a process of analyzing and understanding the data. The goal of EDA is to gain insights into the data, identify patterns, and discover relationships and trends. It helps to identify outliers, missing values, and any other issues that may affect the analysis and modeling of the data.
3. Data Cleaning: Data cleaning is the process of identifying and correcting or removing inaccuracies, inconsistencies, and handling missing values in a dataset. We inspected the dataset for duplicate values. The null value and outlier detection and treatment followed. For the imputation of the null value we used the Mean, Median, and Mode techniques, and for the outliers, we used the Clipping method to handle the outliers without any loss to the data.
4. Feature Selection: At this step, we did the encoding of categorical features. We used the correlation coefficient, encoding, feature manipulation, and feature selection techniques to select the most relevant features. SMOTE is used to address the class imbalance in the target variable.
5. Feature Scaling: We scaled the features to bring down all of the values to a similar range. 
6. Model Training and Implementation: We pass the features to 8 different classification models. We also did hyperparameter tuning using GridSearchCV.
7. Performance Evaluation: After passing it to various classification models and calculating the metrics, we choose a final model that can make better predictions. We evaluated different performance metrics and chose our final model using the f1 score and recall score.

## Artifacts

#### Dataset Source
MongoDB

#### Preprocessings steps
1. Handling Outliers
2. Scaling data
3. Handling imbalance dataset


#### Algorithms used to find best model
1. LogisticRegression
2. SVC
3. RandomForestClassifier
4. GradientBoostingClassifier
5. KNeighborsClassifier
6. DecisionTreeClassifier

#### End Result
* Best Model with params: SVC(C=0.01, degree=5, kernel='poly')
* Training Dataset - Recall Score: : 0.9293388415965623
* Validation Dataset - Recall Score: 0.9144074360960496
* Test Dataset - Recall Score: 0.9322429906542056

## LIVE
* https://credit-card-defaults-prediction.streamlit.app/
 
## Uploading records to MongoDB
```bash
# making connection with mongo db
from pymongo.mongo_client import MongoClient

# Create a new client and connect to the server
uri = <DB URI>
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
    
# Creating DB instances of database which is already created from MongoDB Atlas
db=client["credit_card_defaults"]
collection= db['data']

# inserting the records into mongo db
collection.insert_many(records)

# Retrieve data from the collection
data = list(collection.find())

# Load data into a Pandas DataFrame
df = pd.DataFrame(data)
df.sample(3)
```



