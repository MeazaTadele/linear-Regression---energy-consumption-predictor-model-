import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv('energy_dataset.csv')
print("Shape:", data.shape)
# print(data.isnull().sum())
data = data.dropna()

# print("\nFeatures:", data.columns)

# Check the number of rows before and after cleaning
# print(f"Rows before cleaning: {data.shape[0]}")
# print(f"Rows after cleaning: {data.shape[0]}")


data['date'] = pd.to_datetime(data['date'])
data['hour'] = data['date'].dt.hour
data['day_of_week'] = data['date'].dt.dayofweek
data['month']= data['date'].dt.month

data = data.drop(['date'], axis=1)

# print("\nFeatures:", data.columns)


x = data.drop(columns=['Appliances'])
y = data['Appliances']



x_train , x_temp, y_train , y_temp = train_test_split(x,y,test_size=0.4 , random_state=42)
x_val , x_test, y_val , y_test = train_test_split(x_temp,y_temp , test_size=0.5, random_state=42)



# checking linear regression assumptions

# 1. Linearity

# plt.figure(figsize=(8,6))
# sns.scatterplot(x=x_train['T7'], y=y_train)
# plt.title('Scatter Plot of lights vs Target')
# plt.xlabel('T7')
# plt.ylabel('Appliances')
# plt.show()


# vif_data = pd.DataFrame()
# vif_data["feature"] = x_train.columns
# vif_data["VIF"] = [variance_inflation_factor(x_train.values, i) for i in range(x_train.shape[1])]

# # Display the VIF data
# print(vif_data)