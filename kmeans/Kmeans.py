from PIL import Image
import numpy as np


def kmeans(image_data, num_of_clusters, max_iter, attempts=3):
    data_size = image_data.shape[0]
    attempt_results = []

    for attempt in range(attempts):
        random_pixel_indices = np.random.randint(data_size, size=num_of_clusters)
        centers = np.asarray(image_data[random_pixel_indices, :], dtype=np.float32)
        prev_centers = np.zeros((num_of_clusters, 3))

        clusters = np.zeros(data_size)
        distances = np.zeros((data_size, num_of_clusters))

        cur_iter = 0
        cur_functional = 10_000

        while cur_functional and cur_iter < max_iter:
            prev_centers = np.copy(centers)

            for i in range(num_of_clusters):
                distances[:, i] = np.linalg.norm(image_data - centers[i], axis=1)
            clusters = np.argmin(distances, axis=1)

            for i in range(num_of_clusters):
                centers[i] = np.mean(image_data[clusters == i], axis=0)

            cur_iter += 1
            cur_functional = np.linalg.norm(centers - prev_centers)

        attempt_results.append((cur_functional, np.asarray(centers, dtype=np.uint8), clusters))

    best_answer = (10_000, [], [])
    for attempt in attempt_results:
        if attempt[0] < best_answer[0]:
            best_answer = attempt
    return best_answer[1], best_answer[2]


""" take those images """
images = ['lena.jpg', 'grain.jpg', 'peppers.jpg']

""" try to divide by those number of clusters """
list_of_clusters = [2, 4, 8]

for img in images:
    image = np.array(Image.open(img))
    X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
    iteration_number = 200

    for cluster_number in list_of_clusters:
        final_centers, final_clusters = kmeans(X, cluster_number, iteration_number)
        new_X = np.vstack([final_centers[i] for i in final_clusters])
        new_image = new_X.reshape(image.shape)
        output_image_file = str.replace(img, '.jpg', '_') \
            + str(cluster_number) + '_clusters.jpg'
        Image.fromarray(new_image).save(output_image_file)
        Image.fromarray(new_image, mode="RGB").save(output_image_file)
