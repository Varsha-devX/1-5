1.


import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris  
iris = load_iris() 
df = pd.DataFrame(data=iris.data, columns=iris.feature_names) 
df['species'] = iris.target 
df['species'] = df['species'].map({i: species for i, species in enumerate(iris.target_names)}) 
print("First 5 rows of dataset:") 
display(df.head()) 
print("\nDataset Info:") 
df.info() 
print("\nSummary Statistics:") 
display(df.describe()) 
sns.pairplot(df, hue="species") 
plt.suptitle("Pairplot of Iris Features by Species", y=1.02) 
plt.show() 
plt.figure(figsize=(8, 6)) 
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm') 
plt.title("Correlation Matrix") 
plt.show() 
for feature in iris.feature_names: 
    plt.figure(figsize=(6, 4)) 
 sns.boxplot(x='species', y=feature, data=df) 
    plt.title(f"{feature.capitalize()} by Species") 
    plt.show() 

2.

import pandas as pd 
import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
iris = load_iris() 
X = pd.DataFrame(iris.data, columns=iris.feature_names) 
np.random.seed(0) 
X.loc[X.sample(frac=0.1).index, 'sepal length (cm)'] = np.nan  
X['flower_type'] = np.random.choice(['Type A', 'Type B', 'Type C'], size=len(X)) 
imputer = SimpleImputer(strategy='mean') 
X_numeric_imputed = imputer.fit_transform(X[iris.feature_names]) 
X_numeric_imputed_df = pd.DataFrame(X_numeric_imputed, columns=iris.feature_names) 
print("After Imputation (Missing Values Handled):") 
print(X_numeric_imputed_df.head())
encoder = OneHotEncoder(handle_unknown='ignore') 
X_categorical_encoded = encoder.fit_transform(X[['flower_type']]) 
encoded_df = pd.DataFrame(X_categorical_encoded.toarray(), 
columns=encoder.get_feature_names_out()) 
 scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X_numeric_imputed_df) 
X_scaled_df = pd.DataFrame(X_scaled, columns=iris.feature_names) 
print("\n After Standardization (Feature Scaling Applied):") 
print(X_scaled_df.head()) 
X_processed = pd.concat([X_scaled_df, encoded_df], axis=1) 
print("\n Final Preprocessed Dataset:") 
print(X_processed.head()) 
print("\n Final shape:", X_processed.shape) 


3.

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
iris = datasets.load_iris() 
X = iris.data 
y = iris.target 
target_names = iris.target_names 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) 
k_values = list(range(1, 21)) 
accuracies = [] 
 
for k in k_values: 
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_train, y_train) 
    y_pred = knn.predict(X_test) 
    acc = accuracy_score(y_test, y_pred) 
    accuracies.append(acc) 
    print(f"k={k}: Accuracy = {acc:.2f}") 
plt.figure(figsize=(10, 6)) 
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b') 
plt.title("k-NN Accuracy on Iris Dataset") 
plt.xlabel("Number of Neighbors (k)") 
plt.ylabel("Accuracy") 
plt.grid(True) 
plt.show() 
  

4.

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_california_housing 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.preprocessing import StandardScaler 
housing = fetch_california_housing() 
X = pd.DataFrame(housing.data, columns=housing.feature_names) 
y = pd.Series(housing.target, name='MedHouseVal') 

X_selected = X[['MedInc', 'HouseAge', 'AveRooms']] 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X_selected) 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  
model = LinearRegression() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred)) 
print("R² Score:", r2_score(y_test, y_pred)) 
 plt.figure(figsize=(8, 6)) 
plt.scatter(y_test, y_pred, edgecolors='k', alpha=0.7) 
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--') 
plt.xlabel("Actual Prices") 
plt.ylabel("Predicted Prices") 
plt.title("California Housing - Actual vs Predicted Prices")  
plt.grid(True) 
plt.show() 



5.

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
import matplotlib.pyplot as plt 
import seaborn as sns 
np.random.seed(42) 
n_samples = 1000 
age = np.random.randint(18, 70, n_samples) 
income = np.random.randint(20000, 150000, n_samples) 
browsing_time = np.random.normal(5, 2, n_samples)  
clicked = (0.3 * (age < 30) + 0.4 * (income < 50000) + 0.5 * (browsing_time > 5) + 
           np.random.normal(0, 0.2, n_samples)) > 0.8 
clicked = clicked.astype(int) 
 df = pd.DataFrame({ 
    'Age': age, 
    'Annual_Income': income, 
    'Browsing_Hours': browsing_time, 
    'Clicked_on_Ad': clicked 
}) 
X = df[['Age', 'Annual_Income', 'Browsing_Hours']] 
y = df['Clicked_on_Ad'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
model = LogisticRegression() 
model.fit(X_train, y_train)  
y_pred = model.predict(X_test) 
print("Confusion Matrix:") 
print(confusion_matrix(y_test, y_pred)) 
Print("\nClassification  Report")
print(classification_report(y_test, y_pred)) 
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))


