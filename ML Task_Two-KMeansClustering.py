'''
TASK-2:
    Create a K-means clustering algorithm to group customers
    of a retail store based on their purchase history.
'''

# Importing necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# DATA COLLECTION
# Load the dataset
df = pd.read_csv("D:\Internship\Prodigy InfoTech\Mall_Customers_edit.csv")
print("Sample of raw dataset:")
print(df.head())
print()


# Checking the type of datatypes
print("Datatypes:\n",df.dtypes)



# DATA PREPROCESSING
print("\nChecking for missing values:\n",df.isnull().sum())

# Handling missing values
df = df.dropna()                                    #Removes the rows with missing values

print("\nHandling the missing values:\n",df)

# Checking for duplicate values
print('\nChecking for duplicate values:\n',df.duplicated())
print()


# Basic Scatter Plot
plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'])
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Raw Scatter Plot')
plt.show()



# Finding K value using ELBOW plot concept
k_range=range(1,10)
SSE=[]
for k in k_range:
    kmeans=KMeans(n_clusters=k, n_init=10)
    kmeans.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    SSE.append(kmeans.inertia_)
print("Sum of Squared Error:\n",SSE)


plt.xlabel('K')
plt.ylabel('Sum of Squared Error')
plt.title('Elbow Plot Method')
plt.plot(k_range,SSE, marker='o', markerfacecolor='blue', markersize=7)
plt.show()

print("\nThe value of K=5, ie, 5 Clusters\n")


# DATA TRANSFORMATION
scaler=MinMaxScaler()               #Adjusting the scaling of the axis
scaler.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
df[['Annual Income (k$)','Spending Score (1-100)']]=scaler.transform(df[['Annual Income (k$)','Spending Score (1-100)']])
print('Transformed Data:')
print(df)

# Predicting and fit the data points into Clusters
kmeans=KMeans(n_clusters=5, n_init=10)
y_predicted=kmeans.fit_predict(df[['Annual Income (k$)','Spending Score (1-100)']])
print('The data points belong to the following clusters:')
print(y_predicted)
df['Cluster']=y_predicted
print()
print(df.head())


# Centroid Coordinates
print('\nCo-ordinates of the 5 centroids are:')
print(kmeans.cluster_centers_)

# DATA VISUALIZATION
# Grouping the Clusters
df1=df[df.Cluster==0]
df2=df[df.Cluster==1]
df3=df[df.Cluster==2]
df4=df[df.Cluster==3]
df5=df[df.Cluster==4]
plt.scatter(df1['Annual Income (k$)'],df1['Spending Score (1-100)'],color='purple', label='Cluster 1')
plt.scatter(df2['Annual Income (k$)'],df2['Spending Score (1-100)'],color='blue', label='Cluster 2')
plt.scatter(df3['Annual Income (k$)'],df3['Spending Score (1-100)'],color='green', label='Cluster 3')
plt.scatter(df4['Annual Income (k$)'],df4['Spending Score (1-100)'],color='magenta', label='Cluster 4')
plt.scatter(df5['Annual Income (k$)'],df5['Spending Score (1-100)'],color='orange', label='Cluster 5')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(
kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1,], color='red',marker='*',label='Centroid')
plt.title('K-Means Clustering')
plt.legend()
plt.show()






