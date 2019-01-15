from PIL import Image
import numpy as np
import kmeansAndMetrics
from sys import maxsize
import matplotlib.pyplot as plt

cluster_range = range(2, 10)
number_of_tries = len(cluster_range)
iteration_number = 200

# inner criteria------------------------------------------------------------------------------


davies_bouldin_results = []
calinski_harabaz_results = []


for cluster_number in cluster_range:
    image = np.array(Image.open('policemen.jpg'))
    X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
    centers, closest_clusters = kmeansAndMetrics.kmeans(X, cluster_number, iteration_number)

    davies_bouldin = kmeansAndMetrics.davies_bouldin_index(X, centers, closest_clusters, cluster_number)
    davies_bouldin_results.append(davies_bouldin)

    calinski_harabaz = kmeansAndMetrics.calinski_harabaz_index(X, centers, closest_clusters, cluster_number)
    calinski_harabaz_results.append(calinski_harabaz)


best_delta_k = maxsize
best_calinski_harabaz_cluster_number = -1

# count delta_k in Calinski-Harabaz index for (3..9) clusters
for index in range(1, number_of_tries - 1):
    delta_k = calinski_harabaz_results[index + 1] \
              - 2 * calinski_harabaz_results[index] \
              + calinski_harabaz_results[index - 1]

    if delta_k < best_delta_k:
        best_calinski_harabaz_cluster_number = index
        best_delta_k = delta_k

# remember about index shift, best_cluster_number = best_index + 2
best_calinski_harabaz_cluster_number += 2
best_davies_bouldin_cluster_number = np.argmin(davies_bouldin_results) + 2

image = np.array(Image.open('policemen.jpg'))
X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
iteration_number = 200
best_cluster_number = (best_calinski_harabaz_cluster_number + best_davies_bouldin_cluster_number) // 2
centers, closest_clusters = kmeansAndMetrics.kmeans(X, best_cluster_number, iteration_number)

new_X = np.vstack([centers[i] for i in closest_clusters]).astype(np.uint8)
new_image = new_X.reshape(image.shape)
output_image_file = str(best_cluster_number) + '_cluster_policemen_out.jpg'
Image.fromarray(new_image).save(output_image_file)
Image.fromarray(new_image, mode="RGB").save(output_image_file)


# outer criteria------------------------------------------------------------------------------


outer_criteria_data = 'outer_criteria_data.txt'
data = np.loadtxt(outer_criteria_data, delimiter=' ')
real_clusters, points = data[:, 0], data[:, 1:]

best_rand_statistic_cluster_number = 0
best_fowlkes_mallows_cluster_number = 0
rand_stat_results = []

best_rand_statistic_value = 0
best_fowlkes_mallows_value = 0
fowlkes_mallows_resuts = []

for cluster_number in cluster_range:
    centers, closest_clusters = kmeansAndMetrics.kmeans(points, cluster_number, iteration_number)
    cur_rand_stat, cur_fowlkes_mallows = kmeansAndMetrics.rand_stat_fowlkes_mallows(closest_clusters, real_clusters)

    rand_stat_results.append(cur_rand_stat)
    fowlkes_mallows_resuts.append(cur_fowlkes_mallows)

    if cur_rand_stat > best_rand_statistic_value:
        best_rand_statistic_value = cur_rand_stat
        best_rand_statistic_cluster_number = cluster_number
    if cur_fowlkes_mallows > best_fowlkes_mallows_value:
        best_fowlkes_mallows_value = cur_fowlkes_mallows
        best_fowlkes_mallows_cluster_number = cluster_number


fig, ax = plt.subplots(nrows=2, ncols=2)
ax1, ax2, ax3, ax4 = ax.flatten()

x_axes = list(cluster_range)

ax1.scatter(x=x_axes, y=davies_bouldin_results)
ax1.set_title('Davies-Bouldin. Optimal $k$ is %d' % best_davies_bouldin_cluster_number)

ax2.scatter(x=x_axes, y=calinski_harabaz_results)
ax2.set_title('Calinski-Harabaz. Optimal $k$ is %d' % best_calinski_harabaz_cluster_number)

ax3.scatter(x=x_axes, y=rand_stat_results)
ax3.set_title('Rand Statistic. Optimal $k$ is %d' % best_rand_statistic_cluster_number)

ax4.scatter(x=x_axes, y=fowlkes_mallows_resuts)
ax4.set_title('Fowlkes-Mallows. Optimal $k$ is %d' % best_fowlkes_mallows_cluster_number)

plt.show()
