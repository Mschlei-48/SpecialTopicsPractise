import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data=pd.read_csv("sales_data.csv",sep=",")
print(data.head())
print(data.info())
#Check for missing values
print(data.isnull().sum())
#Check for duplicate values
dup=data.duplicated()
print(dup)
#Drop the duplicated values
print("Shape of data before dropping duplicates :",data.shape)
data=data.drop_duplicates()
print("Shape of data after dropping duplicates :",data.shape)
#extract the date and time in the Order Date column
#data["Date"]=data.loc[:,"Order Date"][0:10]
#(data.head())
data.drop(columns=["Product","catégorie","Purchase Address","Order Date"],inplace=True)
#for i in range(len(data)):
#    data.loc[i,"Date"]=data.loc[i,"Order Date"][0:10]
#for i in range(len(data)):
#    data.loc[i,"Time"]=data.loc[i,"Order Date"][10:]
#data.drop(columns=["Order Date"],inplace=True)
print(data.head())
#Change categorical variables to be numerical variables
#data=pd.get_dummies(data,columns=["Product","catégorie","Purchase Address"])

#Check if all columns are numerical
print(data.info())
#extract the target variable
target=data.loc[:,"margin"]
features=data.iloc[:,:-1]
print("Features are:",features.head())
print("Target variable is:",target.head())
#Split data into train and test
x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.3)
print(x_train.shape, x_test.shape)
#Scale the data
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train,y_train)
x_test=scaler.fit_transform(x_test,y_test)

#Model
model=RandomForestRegressor()
model.fit(x_train,y_train)
#Make predictions
y_pred=model.predict(x_test)
#Get r2score and rmse_score
r2scores=metrics.r2_score(y_test,y_pred)
rmsescores=metrics.mean_squared_error(y_test,y_pred)
with open("results.txt","w") as f:
    f.write(f"R2-Score: {r2scores} \n RMSE-Score: {rmsescores}")
plt.scatter(y_test,y_pred,c="blue")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
plt.savefig("ActualvsPredicted.png")
