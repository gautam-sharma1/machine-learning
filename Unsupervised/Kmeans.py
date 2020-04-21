import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from scipy.spatial import distance
import seaborn as sns
from yellowbrick.cluster.elbow import kelbow_visualizer
from yellowbrick.datasets.loaders import load_nfl
from scipy.spatial.distance import cdist


#IMPORTANT : When figure window pops up, please close it too let the code proceed further.
# There is also one blank figure window generated. Please close it and ignore that. That does not represents any figure.

"""
Please change the value of NUMBER to get different number of clusters
"""
NUMBER = 6
COLOR = ['r', 'g', 'b','c','m','gold','lime','slategrey','orange','orchid']
c_legend = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Clutser 7', 'Clutser 8',
          "Cluster 9",'Cluster 10']

"""
Reading data set and performing normalization
"""
dataset=pd.read_csv('data2.csv',names=["Female","Male", "Age", "Annual income", "Spending score"])
dataset.describe()
x = dataset.iloc[:, [0,1,2, 3,4]].values
X = StandardScaler().fit_transform(x)
"""
Getting optimal value of clusters using elbow method
"""

kelbow_visualizer(KMeans(random_state=0), x, k=(4,12))
wcss = []
for i in range(4,12,2):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


plt.plot(range(4,12,2), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
plt.savefig("elbow.png")



"""
PCA analysis initiated
"""
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(principalComponents)



"""
Starting K-means
"""
kmeans = KMeans(n_clusters=NUMBER , init='k-means++', max_iter=300,  random_state=0)
pred_y = kmeans.fit_predict(principalComponents)
labels = kmeans.fit_predict(principalComponents)
np.savetxt("labels.csv", labels, delimiter=",", fmt='%d', header="Labels")



fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1)

for label, color in zip(range(NUMBER), COLOR[0:NUMBER]):

    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title(str(NUMBER) + " clusters using K-means")
    plt.scatter(principalComponents[labels == label, 0], principalComponents[labels == label, 1], c=color)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow')


plt.show()


"""
Raw Data plots
"""

plt.figure(1 , figsize = (15 , 6))
n = 0
for x in ["Female","Male", "Age", "Annual income", "Spending score"]:
    n += 1
    plt.subplot(1 , 5 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.distplot(dataset[x] , bins = 20)
    plt.title('Distplot of {}'.format(x))
plt.show()


plt.figure(1 , figsize = (15 , 7))
n = 0
for x in ["Female","Male", "Age", "Annual income", "Spending score"]:
    for y in ["Female","Male", "Age", "Annual income", "Spending score"]:
        if x != y:
            n += 1
            plt.subplot(4 , 5 , n)
            plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
            sns.regplot(x = x , y = y , data = dataset)
            plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show()


