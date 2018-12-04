from PIL import Image
import numpy as np


def kmeans(data, clusters, iters):

    def get_closest_cluster(pixel_data):
        nonlocal clusters
        nonlocal centers
        best_distance = 450  # max norm can not be more than this value
        best_center = 0
        for i in range(clusters):
            cur_distance = np.linalg.norm(centers[i] - pixel_data)
            if cur_distance < best_distance:
                best_distance = cur_distance
                best_center = i
        return best_center

    centers = np.random.randint(0, 255, size=(clusters, 3), dtype=np.uint8)
    prev_centers = np.ones((clusters, 3)) * -1
    iter_count = 0

    while iter_count != iters and not np.array_equal(centers, prev_centers):
        prev_centers = np.copy(centers)
        pixels_in_cluster = [list() for _ in range(clusters)]

        # evaluate nearest center for each pixel
        for pixel in data:
            pixels_in_cluster[get_closest_cluster(pixel)].append(pixel)

        # recount centers of clusters
        # indices and dimensions of pixels_in_cluster and centers are same
        for i, cur_pixels_in_cluster in enumerate(pixels_in_cluster):
            sum_of_pixels = np.zeros(3)
            for pixel in cur_pixels_in_cluster:
                sum_of_pixels += pixel
            if cur_pixels_in_cluster:
                centers[i] = sum_of_pixels / len(cur_pixels_in_cluster)

        iter_count += 1

    clustered_data = np.empty(data.shape, dtype=np.uint8)
    for position, pixel in enumerate(data):
        clustered_data[position] = centers[get_closest_cluster(pixel)]

    return centers, clustered_data


""" take those images """
images = ['lena.jpg', 'grain.jpg', 'peppers.jpg']

""" try to divide by those number of clusters """
list_of_clusters = [2, 4, 8]

""" for each (image, number of clusters) 3 attempts """
number_of_attempts = 3

for img in images:
    image = np.array(Image.open(img), dtype=np.uint8)
    X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))

    for cluster_number in list_of_clusters:
        for try_number in range(number_of_attempts):
            res_centers, new_X = kmeans(X, cluster_number, 200)
            new_image = new_X.reshape(image.shape)
            output_image_file = 'output_' + img + '_' \
                                + str(cluster_number) + '_clusters_' \
                                + str(try_number + 1) + '_attempt_' + '.jpg'
            Image.fromarray(new_image, mode="RGB").save(output_image_file)
