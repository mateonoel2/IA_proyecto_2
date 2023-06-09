from PIL import Image
import os
import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from fer import FER
import cv2
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

path = 'Emociones_Dataset'
folder_paths = os.listdir(path)

folder = []

for folder_path in folder_paths:
    print(folder_path)
    folder_path = os.path.join( path, folder_path)
    image_paths = os.listdir(folder_path)

    images = []
    for image_path in image_paths:
        img = Image.open(os.path.join( folder_path, image_path))
        images.append(img)

    folder.append(images)

    """#Asignación de vectores característicos

## LH array
"""

#Load feature vector:
def get_vector(original):
  original = original.resize((224, 224)) 
  coeffs2 = pywt.dwt2(original, 'bior1.3')
  LL, (LH, HL, HH) = coeffs2

  return LL, LH, HL, HH

#Cargamos el feature vector LH, reconoce mejor expreciones faciales
LH_list = []
emocion_list = []

for i in range(len(folder)):
  for img in folder[i]:
    _, LH, _, _ = get_vector(img)
    LH_list.append(LH.flatten())
    emocion_list.append(i)
  
LH_array = np.array(LH_list)
emociones = np.array(emocion_list)

emotion_titles = ['happy', 'contempt', 'fear', 'surprise', 'sadness', 'anger', 'disgust']
_ , counts = np.unique(emociones, return_counts=True)

for value, count in zip(emotion_titles, counts):
    print(f"{value}: {count}")

"""Mostramos imágenes"""

# Load image
img = folder[0][0]
img = img.resize((224,224))
original = img

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']

coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show(block=False)

"""## emotion_vector (fer)

Detectamos emociones base usando fer:
"""

img = folder[0][0] 
img = img.resize((224,224))
img = np.array(img)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

plt.axis('off')
plt.imshow(img)
plt.show(block=False)

detector = FER()

faces = []
for i in range(len(folder)):
    for img in folder[i]:
        img = img.resize((224,224))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        emotions = detector.detect_emotions(img)
        if (emotions == None):
          face = {'emotions': [0,0,0,0,0,0,0]} 
          emotions = [face]
        faces.append(emotions)

for face in faces[:10]:
  print(face)

emotion_vector = []

for face in faces:
  if len(face) >= 1:
    for a in face:
      emotions = a['emotions']
      emotion_vector.append(list(emotions.values()))
      break
  else:
    emotion_vector.append(list(emotions.values()))
  
emotion_vector = np.array(emotion_vector)

for emotion in emotion_vector[:10]:
    print(emotion)

"""# Formula para obtener purezas"""

def get_purity_print(labels):
  emotion_dic = {'happy': labels[:207], 'contempt': labels[207:261], 'fear': labels[261:336],
        'surprise': labels[336:585], 'sadness': labels[585:669], 'anger': labels[669:804], 'disgust': labels[804:981]}

  emotion_accs = []
  for emotion, emotion_labels in emotion_dic.items():
    emotion_values, emotion_counts = np.unique(emotion_labels, return_counts=True)

    print(f'Emotion: {emotion}')
    print(emotion_values)
    print(emotion_counts)

    emotion_sum = np.sum(emotion_counts)
    max_index = np.argmax(emotion_counts)
    max_value = emotion_values[max_index]
    max_count = emotion_counts[max_index]
    emotion_acc = (max_count / emotion_sum) * 100
    print(f'More frecuent label: {max_value}' )
    print(f'{max_count} out of {emotion_sum} recognize the emotion as {emotion}')
    print(f'{emotion} purity: {emotion_acc}')
    print()
    emotion_accs.append(emotion_acc)
  
  emotion_accs = np.array(emotion_accs)
  
  return np.mean(emotion_accs)

def get_purity(labels):
  emotion_dic = {'happy': labels[:207], 'contempt': labels[207:261], 'fear': labels[261:336],
        'surprise': labels[336:585], 'sadness': labels[585:669], 'anger': labels[669:804], 'disgust': labels[804:981]}

  emotion_accs = []
  for emotion, emotion_labels in emotion_dic.items():
    emotion_values, emotion_counts = np.unique(emotion_labels, return_counts=True)
    
    emotion_sum = np.sum(emotion_counts)
    max_index = np.argmax(emotion_counts)
    max_value = emotion_values[max_index]
    max_count = emotion_counts[max_index]
    emotion_acc = (max_count / emotion_sum) * 100
    emotion_accs.append(emotion_acc)
  
  emotion_accs = np.array(emotion_accs)
  
  return np.mean(emotion_accs)


"""# K-means

## K-means with pca
"""

pca_num = 185
clusters = 7

num_dimensions = pca_num
pca = PCA(n_components=num_dimensions, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
data = reduced_LH

# Create a K-means clustering model
k = clusters
kmeans = KMeans(n_clusters=k, random_state=0, n_init = 'auto')

# Fit the K-means model to the data
kmeans.fit(data)

# Predict the cluster labels for each data point
labels = kmeans.predict(data)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity(labels)
print("Average purity:", acc)

"""Luego realizamos GridSearch Basándonos en ARI

* ARI = 1: Concordancia perfecta entre labels y true labels
* ARI = 0: Labels son independientes de true labels
"""

#Parameters: clusters components

max_ari = -1
opt_clusters = -1
opt_components = -1

# Grid Search
for components in range(1, 21): # Component range [1 - 20]

  pca = PCA(n_components=components, random_state=0)
  reduced_LH = pca.fit_transform(LH_array)
  data = reduced_LH

  for clusters in range(2, 11): # Cluster range [2 - 10]
    # Fit the GMM model
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init = 'auto')
    kmeans.fit(data)

    # Predict the labels
    labels = kmeans.predict(data)
    
    # Compute ARI score
    ari = adjusted_rand_score(emociones, labels)

    # Update maximum ARI score and optimal parameters
    if ari > max_ari:
        max_ari = ari
        opt_clusters = clusters
        opt_components = components

  print(f"{components} components out of 20")

print(f"Optimal number of components: {opt_components}")
print(f"Optimal number of clusters: {opt_clusters}")
print(f"Maximum ARI score: {max_ari}\n")

# Optimal number of components: 6
# Optimal number of clusters: 5
# Maximum ARI score: 0.290422025042182

pca_num = 16
clusters = 5

pca = PCA(n_components = pca_num, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
data = reduced_LH

kmeans = KMeans(n_clusters=clusters, random_state=0, n_init = 'auto')

kmeans.fit(data)

labels = kmeans.predict(data)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity(labels)
print("Average purity:", acc)

"""Obtenemos que el modelo más efectivo cuenta con 16 componentes de PCA y 5 clusters.

Usaremos el método elbow para optimizar las características internas del cluster
"""

# Define the range of cluster numbers to explore
min_clusters = 2
max_clusters = 30

sse = [] 

#BIC (Bayesian Information Criterion) or AIC (Akaike Information Criterion) 
for num_clusters in range(min_clusters, max_clusters+1):
    kmeans = KMeans(n_clusters= num_clusters, random_state=0, n_init = 'auto')
    kmeans.fit(data)  # data represents your image dataset
    sse.append(kmeans.inertia_)

plt.figure()
plt.plot(range(min_clusters, max_clusters+1), sse, label='SSE')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')  
plt.title('The Elbow Method showing the optimal k')
plt.show(block=False)

"""El elbow se encuentra en el cluster 23"""

pca_num = 16
clusters = 6

pca = PCA(n_components = pca_num, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
data = reduced_LH

kmeans = KMeans(n_clusters=clusters, random_state=0, n_init = 'auto')

kmeans.fit(data)

labels = kmeans.predict(data)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity(labels)
print("Average purity:", acc)

opt_clusters = 7

# Compute the silhouette scores for each sample
silhouette_vals = silhouette_samples(data, labels)

y_lower = 7 # For space between silhouette plots
for i in range(opt_clusters):  # opt_clusters is the optimal number of clusters found
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()

    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / opt_clusters)  # Color mapping
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))  # Label the silhouette plots with their cluster numbers in the middle

    y_lower = y_upper + 10  # 10 for the 0 samples

plt.xlabel("The silhouette coefficient values")
plt.ylabel("Cluster label")
plt.title("Silhouette plot for the various clusters")
plt.show(block=False)

"""Nuestro modelo KMeans"""

def kmeans_fit(data, n_clusters):
    np.random.seed(0)
    # Initialize centroids randomly from the data points
    idx = np.random.choice(len(data), n_clusters, replace=False)
    centroids = data[idx, :]
    
    while True:
        # Assign each data point to the closest centroid
        labels = kmeans_predict(data, centroids)

        # Compute new centroids as the mean of the data points in each cluster
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(n_clusters)])

        # If centroids do not change, then algorithm has converged
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids

def kmeans_predict(data, centroids):
    # Calculate the distance from each point to each centroid
    dists = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))

    # Return the index of the closest centroid for each point
    return np.argmin(dists, axis=0)

pca_num = 6

pca = PCA(n_components = pca_num, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
data = reduced_LH

# Create a K-means clustering model
k = 23  # Number of clusters

# Fit the K-means model to the data
centroids = kmeans_fit(data, k)

# Predict the cluster labels for each data point
kmeans_labels = kmeans_predict(data, centroids)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity(labels)
print("Average purity:", acc)

opt_clusters = 23

# Compute the silhouette scores for each sample
silhouette_vals = silhouette_samples(data, labels)

y_lower = 10  # For space between silhouette plots
for i in range(opt_clusters):  # opt_clusters is the optimal number of clusters found
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()

    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / opt_clusters)  # Color mapping
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))  # Label the silhouette plots with their cluster numbers in the middle

    y_lower = y_upper + 10  # 10 for the 0 samples

plt.xlabel("The silhouette coefficient values")
plt.ylabel("Cluster label")
plt.title("Silhouette plot for the various clusters")
plt.show(block=False)

"""#GMM

## GMM with pca
"""

pca_num = 185
clusters = 7

num_dimensions = pca_num
pca = PCA(n_components=num_dimensions, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
data = reduced_LH


gmm = GaussianMixture(n_components = clusters, init_params = 'kmeans', random_state = 0)

gmm.fit(data)

labels = gmm.predict(data)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity(labels)
print("Average purity:", acc)

"""Luego realizamos GridSearch Basándonos en ARI

* ARI = 1: Concordancia perfecta entre labels y true labels
* ARI = 0: Labels son independientes de true labels
"""

#Parameters: clusters components

max_ari = -1
opt_clusters = -1
opt_components = -1

# Grid Search
for components in range(1, 21): # Component range [1 - 20]

  pca = PCA(n_components=components, random_state=0)
  reduced_LH = pca.fit_transform(LH_array)
  data = reduced_LH

  for clusters in range(2, 11): # Cluster range [2 - 10]
    # Fit the GMM model
    gmm = GaussianMixture(n_components=clusters, init_params='kmeans', random_state=0)
    gmm.fit(data)

    # Predict the labels
    labels = gmm.predict(data)
    
    # Compute ARI score
    ari = adjusted_rand_score(emociones, labels)

    # Update maximum ARI score and optimal parameters
    if ari > max_ari:
        max_ari = ari
        opt_clusters = clusters
        opt_components = components

  print(f"{components} components out of 20")

print(f"Optimal number of components: {opt_components}")
print(f"Optimal number of clusters: {opt_clusters}")
print(f"Maximum ARI score: {max_ari}\n")

# Optimal number of components: 10
# Optimal number of clusters: 6
# Maximum ARI score: 0.35431966881360755

pca_num = 10
clusters = 6

pca = PCA(n_components = pca_num, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
data = reduced_LH

gmm = GaussianMixture(n_components = clusters, init_params = 'kmeans', random_state = 0)

gmm.fit(data)

labels = gmm.predict(data)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity(labels)
print("Average purity:", acc)

"""Obtenemos que el modelo más efectivo cuenta con 10 componentes de PCA y 6 clusters.

El método elbow para optimizar las características internas del cluster
"""

# Define the range of cluster numbers to explore
min_clusters = 2
max_clusters = 15

bic = []
aic = []

#BIC (Bayesian Information Criterion) or AIC (Akaike Information Criterion) 
for num_clusters in range(min_clusters, max_clusters+1):
    gmm = GaussianMixture(n_components=num_clusters, random_state=0)
    gmm.fit(data)  # data represents your image dataset
    bic.append(gmm.bic(data))
    aic.append(gmm.aic(data))


plt.plot(range(min_clusters, max_clusters+1), bic, label='BIC')
plt.plot(range(min_clusters, max_clusters+1), aic, label='AIC')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Method with GMM')
plt.legend()
plt.show(block=False)

"""Con el método elbow observamos que en el cluster 12 se comienza a estar flat la función

Mostramos los resultados, según la clusterización del método elbow.
"""

pca_num = 10
clusters = 12

pca = PCA(n_components = pca_num, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
data = reduced_LH

gmm = GaussianMixture(n_components = clusters, init_params = 'kmeans', random_state = 0)

gmm.fit(data)

labels = gmm.predict(data)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity(labels)
print("Average purity:", acc)

opt_clusters = 12

# Compute the silhouette scores for each sample
silhouette_vals = silhouette_samples(data, labels)

y_lower = 10  # For space between silhouette plots
for i in range(opt_clusters):  # opt_clusters is the optimal number of clusters found
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()

    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / opt_clusters)  # Color mapping
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))  # Label the silhouette plots with their cluster numbers in the middle

    y_lower = y_upper + 10  # 10 for the 0 samples

plt.xlabel("The silhouette coefficient values")
plt.ylabel("Cluster label")
plt.title("Silhouette plot for the various clusters")
plt.show(block=False)

"""Nuestro modelo GMM"""

from scipy.stats import multivariate_t

def EM(means, covariances, weights, X):
  tol=0.001
  reg_covar=1e-06
  for _ in range(100):
      # E-step: Calculate the responsibilities
      responsibilities = []
      for i in range(X.shape[0]):
          probs = [weights[j] *  multivariate_t.pdf(X[i], means[j], covariances[j], allow_singular=True) for j in range(n_components)]
          probs_sum = np.sum(probs)
          resps = [prob / probs_sum if prob != 0 else 0 for prob in probs]
          responsibilities.append(resps)
      responsibilities = np.array(responsibilities)

      # M-step: Update the model parameters
      for j in range(n_components):
          resp_sum = np.sum(responsibilities[:, j])
          if resp_sum != 0:
              means[j] = np.sum(responsibilities[:, j].reshape(-1, 1) * X, axis=0) / resp_sum
              covariances[j] = np.dot((responsibilities[:, j].reshape(-1, 1) * (X - means[j])).T, X - means[j]) / resp_sum
              covariances[j] += reg_covar * np.eye(X.shape[1])  # Add covariance regularization term
              weights[j] = resp_sum / X.shape[0]

      # Check for convergence
      if np.abs(np.sum(responsibilities) - 1) < tol:
          break

  return responsibilities


def gmm_predict(means, covariances, weights, data):
  responsibilities = EM(means, covariances, weights, data)
  labels = np.argmax(responsibilities, axis=1)
  return responsibilities, labels

pca_num = 10

pca = PCA(n_components = pca_num, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
joined_vector = np.concatenate((emotion_vector, reduced_LH), axis=1)
data = joined_vector

# Create a Gaussian Mixture Model
n_components = 12  
 
kmeans = KMeans(n_clusters=n_components, n_init=1, random_state=0)
kmeans.fit(data)

means = kmeans.cluster_centers_
covariances = np.array([np.eye(data.shape[1]) for _ in range(n_components)])
weights = np.ones(n_components) / n_components

gmm = {
    'means': means,
    'covariances': covariances,
    'weights': weights
}
responsibilities, labels =  gmm_predict(means, covariances, weights, data)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity(labels)
print("Average purity:", acc)

opt_clusters = 12

# Compute the silhouette scores for each sample
silhouette_vals = silhouette_samples(data, labels)

y_lower = 10  # For space between silhouette plots
for i in range(opt_clusters):  # opt_clusters is the optimal number of clusters found
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()

    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / opt_clusters)  # Color mapping
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))  # Label the silhouette plots with their cluster numbers in the middle

    y_lower = y_upper + 10  # 10 for the 0 samples

plt.xlabel("The silhouette coefficient values")
plt.ylabel("Cluster label")
plt.title("Silhouette plot for the various clusters")
plt.show(block=False)

"""## GMM with fer + pca

Empezamos evaluando el modelo base:
"""

pca_num = 4
clusters = 7

num_dimensions = pca_num
pca = PCA(n_components=num_dimensions, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
joined_vector = np.concatenate((emotion_vector, reduced_LH), axis=1)
data = joined_vector

n_components = clusters
gmm = GaussianMixture(n_components=n_components, init_params = 'kmeans', random_state = 0)

gmm.fit(data)

labels = gmm.predict(data)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity(labels)
print("Average purity:", acc)

"""Luego realizamos GridSearch Basándonos en ARI

*   ARI = 1: Concordancia perfecta entre labels y true labels
*   ARI = 0: Labels son independientes de true labels



"""

#Parameters: clusters components

max_ari = -1
opt_clusters = -1
opt_components = -1

# Grid Search
for components in range(1, 21): # Component range [1 - 20]

  pca = PCA(n_components=components, random_state=0)
  reduced_LH = pca.fit_transform(LH_array)
  joined_vector = np.concatenate((emotion_vector, reduced_LH), axis=1)
  data = joined_vector

  for clusters in range(2, 11): # Cluster range [2 - 10]
    # Fit the GMM model
    gmm = GaussianMixture(n_components=clusters, init_params='kmeans', random_state=0)
    gmm.fit(data)

    # Predict the labels
    labels = gmm.predict(data)
    
    # Compute ARI score
    ari = adjusted_rand_score(emociones, labels)

    # Update maximum ARI score and optimal parameters
    if ari > max_ari:
        max_ari = ari
        opt_clusters = clusters
        opt_components = components

    print(f"{components} components out of 20")

print(f"Optimal number of components: {opt_components}")
print(f"Optimal number of clusters: {opt_clusters}")
print(f"Maximum ARI score: {max_ari}\n")

# Optimal number of components: 6
# Optimal number of clusters: 5
# Maximum ARI score: 0.6176227114080428

pca_num = 6
clusters = 5

pca = PCA(n_components = pca_num, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
joined_vector = np.concatenate((emotion_vector, reduced_LH), axis=1)
data = joined_vector

gmm = GaussianMixture(n_components = clusters, init_params = 'kmeans', random_state = 0)

gmm.fit(data)

labels = gmm.predict(data)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity_print(labels)
print("Average purity:", acc)

opt_clusters = 6

# Compute the silhouette scores for each sample
silhouette_vals = silhouette_samples(data, labels)

y_lower = 10  # For space between silhouette plots
for i in range(opt_clusters):  # opt_clusters is the optimal number of clusters found
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()

    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / opt_clusters)  # Color mapping
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))  # Label the silhouette plots with their cluster numbers in the middle

    y_lower = y_upper + 10  # 10 for the 0 samples

plt.xlabel("The silhouette coefficient values")
plt.ylabel("Cluster label")
plt.title("Silhouette plot for the various clusters")
plt.show(block=False)

"""Obtenemos que el modelo más efectivo cuenta con 6 componentes de PCA y 5 clusters.

Ahora utilizaremos sillhuete y el método elbow para optimizar las características internas del cluster
"""

#Elbow method

min_clusters = 2
max_clusters = 10

bic = []
aic = []

#BIC (Bayesian Information Criterion) or AIC (Akaike Information Criterion) 
for num_clusters in range(min_clusters, max_clusters+1):
    gmm = GaussianMixture(n_components=num_clusters, random_state=0)
    gmm.fit(data)
    bic.append(gmm.bic(data))
    aic.append(gmm.aic(data))


plt.plot(range(min_clusters, max_clusters+1), bic, label='BIC')
plt.plot(range(min_clusters, max_clusters+1), aic, label='AIC')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Method with GMM')
plt.legend()
plt.show(block=False)

"""l método elbow indica que 7 es la mejor cantidad de clusters mientras el silhouette score indica que 2.

Reesultados con mejores valores indicados por el método elbow
"""

pca_num = 6
clusters = 7

pca = PCA(n_components = pca_num, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
joined_vector = np.concatenate((emotion_vector, reduced_LH), axis=1)
data = joined_vector

gmm = GaussianMixture(n_components = clusters, init_params = 'kmeans', random_state = 0)

gmm.fit(data)

labels = gmm.predict(data)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity_print(labels)
print("Average purity:", acc)

opt_clusters = 7

# Compute the silhouette scores for each sample
silhouette_vals = silhouette_samples(data, labels)

y_lower = 10  # For space between silhouette plots
for i in range(opt_clusters):  # opt_clusters is the optimal number of clusters found
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()

    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / opt_clusters)  # Color mapping
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))  # Label the silhouette plots with their cluster numbers in the middle

    y_lower = y_upper + 10  # 10 for the 0 samples

plt.xlabel("The silhouette coefficient values")
plt.ylabel("Cluster label")
plt.title("Silhouette plot for the various clusters")
plt.show(block=False)

"""Nuestra GMM con fer + PCA (Comparamos el modelo final con nuestra implementación del algoritmo) 


"""

pca_num = 6

pca = PCA(n_components = pca_num, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
joined_vector = np.concatenate((emotion_vector, reduced_LH), axis=1)
data = joined_vector

# Create a Gaussian Mixture Model
n_components = 7  
 
kmeans = KMeans(n_clusters=n_components, n_init=1, random_state=0)
kmeans.fit(data)

means = kmeans.cluster_centers_
covariances = np.array([np.eye(data.shape[1]) for _ in range(n_components)])
weights = np.ones(n_components) / n_components

gmm = {
    'means': means,
    'covariances': covariances,
    'weights': weights
}

responsibilities = EM(means, covariances, weights, data)
labels = np.argmax(responsibilities, axis=1)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity_print(labels)
print("Average purity:", acc)

opt_clusters = 7

# Compute the silhouette scores for each sample
silhouette_vals = silhouette_samples(data, labels)

y_lower = 10  # For space between silhouette plots
for i in range(opt_clusters):  # opt_clusters is the optimal number of clusters found
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()

    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / opt_clusters)  # Color mapping
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))  # Label the silhouette plots with their cluster numbers in the middle

    y_lower = y_upper + 10  # 10 for the 0 samples

plt.xlabel("The silhouette coefficient values")
plt.ylabel("Cluster label")
plt.title("Silhouette plot for the various clusters")
plt.show(block=False)

"""#DBSCAN

## DBSCAN with pca
"""

pca_num = 185

num_dimensions = pca_num
pca = PCA(n_components=num_dimensions, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
data = reduced_LH

# Create a DBSCAN clustering model
eps = 5000 # Maximum distance between two samples to be considered in the same neighborhood
min_samples = 1  # Minimum number of samples required to form a dense region
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')

# Fit the BSCAN model to the data
labels = dbscan.fit_predict(data)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity(labels)
print("Average purity:", acc)

"""Luego realizamos GridSearch Basándonos en ARI

* ARI = 1: Concordancia perfecta entre labels y true labels
* ARI = 0: Labels son independientes de true labels
"""

#Parameters: clusters components

max_ari = -1
opt_clusters = -1
opt_eps = -1

# Grid Search
for components in range(1, 21): # Component range [1 - 20]

  pca = PCA(n_components=components, random_state=0)
  reduced_LH = pca.fit_transform(LH_array)
  data = reduced_LH

  for eps in range(100, 8001, 100): # Eps range [100 - 8000]
    min_samples = 2
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')

    # Predict the labels
    labels = dbscan.fit_predict(data)
    
    # Compute ARI score
    ari = adjusted_rand_score(emociones, labels)

    # Update maximum ARI score and optimal parameters
    if ari > max_ari:
        max_ari = ari
        opt_eps = eps
        opt_components = components

  print(f"{components} components out of 20")

print(f"Optimal number of components: {opt_components}")
print(f"Optimal number of eps: {opt_eps}")
print(f"Maximum ARI score: {max_ari}\n")

pca_num = 6

num_dimensions = pca_num
pca = PCA(n_components=num_dimensions, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
data = reduced_LH

# Create a DBSCAN clustering model
eps = 200 # Maximum distance between two samples to be considered in the same neighborhood
min_samples = 2  # Minimum number of samples required to form a dense region
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')

# Fit the BSCAN model to the data
labels = dbscan.fit_predict(data)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity(labels)
print("Average purity:", acc)

from sklearn.neighbors import NearestNeighbors

min_samples = 2

neighbors = NearestNeighbors(n_neighbors=min_samples, metric='manhattan')
neighbors_fit = neighbors.fit(data)
distances, indices = neighbors_fit.kneighbors(data)

distances = np.sort(distances, axis=0)
distances = distances[:,1]

plt.figure()
plt.plot(distances)
plt.xlabel('Points sorted according to distance of nth nearest neighbor')
plt.ylabel('n-th nearest neighbor distance')  
plt.title('k-distance Graph')
plt.show(block=False)

"""Nuestro DBSCAN"""

class dbscan_noemi:
    def __init__(self, eps, minPts, X):
        self.eps = eps ## Radio. Distancia máxima entre dos puntos para ser considerados vecinos
        self.minPts = minPts ## Densidad minima de puntos para formar una región
        self.clusters = []
        self.visited = set()
        self.tree = None
        self.X = X

    def fit(self, X):
        self.tree = KDTree(X) # se usará KdTree como algoritmo de estructura de datos
        self._dbscan(X)

    def predict(self, X):
        labels = np.full(len(X), -1)  # Inicializo a todos los puntos como ruido (-1)

        for i, point in enumerate(X):
            if tuple(point) not in self.visited:
                continue

            for cluster_label, cluster in enumerate(self.clusters):
                if tuple(point) in cluster:
                    labels[i] = cluster_label
                    break

        return labels

    def _dbscan(self, X):
        for i, point in enumerate(X):
            if tuple(point) not in self.visited:
                self.visited.add(tuple(point))
                neighbors = self._regionQuery(point)

                if len(neighbors) < self.minPts:
                    # marcar a los vecinos como ruido
                    self.clusters.append(set())
                else:
                    cluster = set()
                    self.clusters.append(cluster)
                    self._expandCluster(point, neighbors, cluster)

    def _expandCluster(self, point, neighbors, cluster):
        cluster.add(tuple(point))

        for neighbor in neighbors:
            if tuple(neighbor) not in self.visited:
                self.visited.add(tuple(neighbor))
                newNeighbors = self._regionQuery(neighbor)

                if len(newNeighbors) >= self.minPts:
                    neighbors.extend(newNeighbors)

                if not self._isAssignedToAnyCluster(neighbor):
                    cluster.add(tuple(neighbor))

    def _regionQuery(self, point):
        neighbor_indices = self.tree.query_ball_point(point, self.eps)
        neighbors = [self.X[i] for i in neighbor_indices] 
        return neighbors


    def _isAssignedToAnyCluster(self, point):
        for cluster in self.clusters:
            if tuple(point) in cluster:
                return True
        return False

pca_num = 6

num_dimensions = pca_num
pca = PCA(n_components=num_dimensions, random_state = 0)
reduced_LH = pca.fit_transform(LH_array)
data = reduced_LH

eps = 200  # Adjust the value of eps as needed
min_samples = 2  # Adjust the value of minPts as needed
dbscan = dbscan_noemi(eps=eps, minPts = min_samples, X=data)
dbscan.fit(X = data)

# Predict cluster labels for new data
labels = dbscan.predict(X = data)

print("Internal meassures\n")

dbi = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", dbi)

silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg,'\n')

print("External meassures\n")

ari = adjusted_rand_score(emociones, labels)
print("Adjusted Rand Index (ARI):", ari)

nmi = normalized_mutual_info_score(emociones, labels)
print("Normalized Mutual Information (NMI):", nmi)

acc = get_purity_print(labels)
print("Average purity:", acc)