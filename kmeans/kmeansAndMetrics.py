import numpy as np
from math import sqrt


def kmeans(data, num_of_clusters, max_iter, attempts=3):
    data_size = data.shape[0]
    attempt_results = []

    for attempt in range(attempts):
        random_pixel_indices = np.random.randint(data_size, size=num_of_clusters)
        centers = np.asarray(data[random_pixel_indices, :], dtype=np.float32)
        prev_centers = np.zeros((num_of_clusters, 3))

        closest_clusters = np.zeros(data_size)
        distances = np.zeros((data_size, num_of_clusters))

        cur_iter = 0
        cur_functional = 10_000

        while cur_functional and cur_iter < max_iter:
            prev_centers = np.copy(centers)

            for i in range(num_of_clusters):
                distances[:, i] = np.linalg.norm(data - centers[i], axis=1)
            closest_clusters = np.argmin(distances, axis=1)

            for i in range(num_of_clusters):
                centers[i] = np.mean(data[closest_clusters == i], axis=0)

            cur_iter += 1
            cur_functional = np.linalg.norm(centers - prev_centers)

        attempt_results.append((cur_functional, np.asarray(centers, dtype=np.uint16), closest_clusters))

    best_answer = (10_000, [], [])
    for attempt in attempt_results:
        if attempt[0] < best_answer[0]:
            best_answer = attempt
    return best_answer[1], best_answer[2]


def davies_bouldin_index(data, centers, closest_clusters, num_of_clusters):
    sqrt_variance = np.zeros(num_of_clusters, dtype=np.float32)
    for i in range(num_of_clusters):
        sqrt_variance[i] = sqrt(np.var(data[closest_clusters == i]))

    clusters_distance = np.zeros((num_of_clusters, num_of_clusters), dtype=np.float32)
    for i in range(num_of_clusters):
        for j in range(num_of_clusters - i):
            clusters_distance[i][j] = np.linalg.norm(centers[i] - centers[j])
            clusters_distance[j][i] = clusters_distance[i][j]

    db_between_clusters = np.zeros((num_of_clusters, num_of_clusters), dtype=np.float32)
    for i in range(num_of_clusters):
        for j in range(num_of_clusters - i):
            if i != j:
                db_between_clusters[i][j] = (sqrt_variance[i] + sqrt_variance[j]) / clusters_distance[i][j]
                db_between_clusters[j][i] = db_between_clusters[i][j]
            else:
                # prevent choosing db_between_clusters[i][i] in next step
                db_between_clusters[i][i] = -1

    max_between_clusters = np.max(db_between_clusters, axis=1)
    return np.sum(max_between_clusters) / num_of_clusters


def calinski_harabaz_index(data, centers, closest_clusters, num_of_clusters):
    data_size, _ = data.shape
    within_disp = np.sum([np.sum((data[closest_clusters == i] - centers[i]) ** 2)
                          for i in range(num_of_clusters)])
    mean = np.mean(data, axis=0)
    between_disp = np.sum([len(data[closest_clusters == i]) * ((centers[i] - mean) ** 2)
                           for i in range(num_of_clusters)])

    return between_disp / within_disp * (data_size - num_of_clusters) / (num_of_clusters - 1)


def count_tp_tn_fp_fn(closest_clusters, real_clusters):
    data_size = closest_clusters.shape[0]
    tp, tn, fp, fn = (0., 0., 0., 0.)
    for i in range(data_size):
        for j in range(i + 1, data_size):
            if closest_clusters[i] == closest_clusters[j] and real_clusters[i] == real_clusters[j]:
                tp += 1
            elif closest_clusters[i] != closest_clusters[j] and real_clusters[i] == real_clusters[j]:
                fn += 1
            elif closest_clusters[i] == closest_clusters[j] and real_clusters[i] != real_clusters[j]:
                fp += 1
            else:
                tn += 1
    return tp, tn, fp, fn


def rand_stat_fowlkes_mallows(closest_clusters, real_clusters):
    num_of_clusters = closest_clusters.shape[0]
    tp, tn, fp, fn = count_tp_tn_fp_fn(closest_clusters, real_clusters)
    rand_statistic_index = (tp + tn) / (tp + tn + fp + fn)
    fowlkes_mallows_index = tp / sqrt((tp + fn) * (tp + fp))
    return rand_statistic_index, fowlkes_mallows_index
