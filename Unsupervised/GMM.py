from sklearn.mixture import GaussianMixture as GMMM
from sklearn.decomposition import PCA
import pandas as pd

import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from gmSuper import GMM as GMM


from matplotlib.patches import Ellipse


#IMPORTANT : When figure window pops up, please close it too let the code proceed further.
# There is also one blank figure window generated. Please close it and ignore that. That does not represents any figure.

####################################################
"""
Kindly manipulate this variable to get different clusters. 
"""
NUMBER = 6
####################################################

"""
Following function is used in plotting
"""
####################################################
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
####################################################

"""
Reading data set and performing normalization
"""
from sklearn.preprocessing import LabelEncoder
dataset=pd.read_csv('data2.csv',names=["Female","Male", "Age", "Annual income", "Spending score"])

dataset.describe()
x = dataset.iloc[:, [0,1,2, 3,4]].values
X = StandardScaler().fit_transform(x)
df = pd.DataFrame(x)

"""
Implementing PCA on our data

"""
from sklearn.cluster import KMeans
sklearn_pca = PCA(n_components = 2)
Y_sklearn = sklearn_pca.fit_transform(X)


"""
Plotting log-likelihood. Please take note that due to its probabilistic nature log-likelihood may
sometimes give the optimal value as 8. But most of the times it gives the value of 6, hence we have taken its optimal value
to be 6.
"""

score_ = [] #log score for each cluster


for i in range(4,12,2):
    model = GMMM(i,covariance_type='full',max_iter=100) #GMMM is the function available from sklearn. An extra M is
    # introduced to reduce ambiguity as the external function used also has a name GMM.
    fitted_values = model.fit(Y_sklearn)
    score_.append(model.score(Y_sklearn))


plt.plot([i for i in range(4,12,2)],[k for k in score_], label='log_likelihood')

plt.legend(loc='best')
plt.xlabel('number of clusters')
plt.show()
plt.savefig('log likelihood.png')

""""
GMM model Initialised
"""
model = GMM(NUMBER, 700)
fitted_values = model.fit(Y_sklearn)
predicted_values = model.predict(Y_sklearn)
np.savetxt("labels_gmm.csv", predicted_values, delimiter=",", fmt='%d', header="GMM Labels")

centers = np.zeros((NUMBER, 2))

"""
Plotting PCA results with GMM predicted results
"""
for i in range(model.C):
    density = mvn(cov=model.sigma[i], mean=model.mu[i]).logpdf(Y_sklearn)
    centers[i, :] = Y_sklearn[np.argmax(density)]
fig2=plt.figure()
plt.figure(figsize=(10, 8))
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=predicted_values, s=50, cmap='viridis', zorder=1)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=300, alpha=0.5, zorder=2);
plt.title(str(NUMBER)+" Clusters using EM")

w_factor = 0.2 / model.pi.max()

for pos, covar, w in zip(model.mu, model.sigma, model.pi):
    draw_ellipse(pos, covar, alpha=w)
plt.savefig("GMM_6.png")
plt.show()
