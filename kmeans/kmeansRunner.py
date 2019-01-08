from PIL import Image
import numpy as np
import kmeansAndMetrics

""" take those images """
images = ['lena.jpg', 'grain.jpg', 'peppers.jpg']

""" try to divide by those number of clusters """
list_of_clusters = [2, 4, 8]

for img in images:
    image = np.array(Image.open(img))
    X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
    iteration_number = 200

    for cluster_number in list_of_clusters:
        final_centers, final_closest_clusters = kmeansAndMetrics.kmeans(X, cluster_number, iteration_number)
        print(final_closest_clusters.shape)
        new_X = np.vstack([final_centers[i] for i in final_closest_clusters])
        new_image = new_X.reshape(image.shape)
        output_image_file = str.replace(img, '.jpg', '_') \
            + str(cluster_number) + '_clusters.jpg'
        Image.fromarray(new_image).save(output_image_file)
        Image.fromarray(new_image, mode="RGB").save(output_image_file)
