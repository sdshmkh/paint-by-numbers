import datetime

import cv2
import numpy as np
import skimage
from cv2 import ximgproc
from sklearn.cluster import MiniBatchKMeans

from openai import OpenAI


def record_time(func):
    def wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        res = func(*args, **kwargs)
        end = datetime.datetime.now()
        diff = end-start
        print("{} finished in {:.2f}s".format(func, diff.total_seconds()))
        return res
    return wrapper

@record_time
def simplify_image(image, n_clusters=8):
    im_shape = image.shape
    image = image.reshape(-1, 3)
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    model.fit(image)
    return model, model.cluster_centers_[model.labels_].reshape(im_shape).astype(np.uint8)


@record_time
def apply_slic(img):
    slic = ximgproc.createSuperpixelSLIC(img, algorithm=ximgproc.SLIC, region_size=10, ruler=10.0)
    slic.iterate(30)
    slic.enforceLabelConnectivity()
    labels = slic.getLabels()
    number_of_superpixels = slic.getNumberOfSuperpixels()
    mask = slic.getLabelContourMask()
    image_with_boundaries = cv2.bitwise_and(img, img, mask=~mask)
    return slic, image_with_boundaries, labels, number_of_superpixels

@record_time
def assign_superpixel_to_clusters(sp_img, labels, num_sp, model):
    im_shape = list(sp_img.shape)
    reconstructed_image = np.copy(sp_img)
    reconstructed_image = np.dstack([reconstructed_image, np.ones(im_shape[:2])])
    kmeans_labels = list()
    for i in range(num_sp):
        mask = labels == i
        cluster_labels = model.labels_[mask.flatten()]
        cluster_label = np.bincount(cluster_labels).argmax()
        alpha = 50
        color_with_alpha = np.concatenate([np.round(model.cluster_centers_[cluster_label]), [alpha]])
        reconstructed_image[mask, :] = color_with_alpha
        kmeans_labels.append(cluster_label)
        # reconstructed_image[]
    return reconstructed_image.astype(np.uint8), kmeans_labels


@record_time
def create_image_labels(slic, recon_image, cluster_labels, kmeans_labels):
    num_superpixels = slic.getNumberOfSuperpixels()
    mask = slic.getLabelContourMask()
    image_with_boundaries = cv2.bitwise_and(recon_image, recon_image, mask=~mask)
    image_with_labels = np.copy(image_with_boundaries)
    FONT_SCALE = 2e-2
    for i in range(num_superpixels):
        y, x = np.where(cluster_labels == i)
        
        w = np.abs(x.min() - x.max())
        h = np.abs(y.min() - y.max())
        
        centroid = (int(np.mean(x)), int(np.mean(y)))
        
        font_scale = min(w, h) * FONT_SCALE
        thickness = np.round(min(w, h) * FONT_SCALE).astype(np.uint8)
        cv2.putText(image_with_labels, str(kmeans_labels[i]), centroid, 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0, 50), thickness)
    return image_with_labels

def combine_images(image, pal):
    ims = list(image.shape)
    ims[0] = ims[0] + 200
    combine = np.zeros(ims) + 255
    combine[0:image.shape[0], :, :] = image
    
    
    p_arr = np.zeros((180, image.shape[-2], image.shape[-1])) + 255
    a_y = ims[1]
    cols = pal.shape[0]
    multi =a_y // cols
    for i in range(cols):
        p_arr[:, (i)*(multi):(i+1)*(multi), :] = pal[ i, :]
        loc = (ims[0], ((i)*(multi) + (i+1)*(multi))//2)
        cv2.putText(p_arr, str(i), loc, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0, 50), 1)

    combine[image.shape[0]+20:, :, :] = p_arr
    return combine.astype(np.uint8)

@record_time
def create(image_path, n_clusters=8):
    image = skimage.io.imread(image_path)
    if image.shape[-1] > 3:
        # get rid of the alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    model, simple_image = simplify_image(image, n_clusters)
    palette = np.hstack((model.cluster_centers_, 255 * np.ones((model.cluster_centers_.shape[0], 1), dtype=np.uint8)))

    slic, sp_image, labels, num_superpixels = apply_slic(simple_image)
    recon_image, kmeans_labels = assign_superpixel_to_clusters(sp_image, labels, num_superpixels, model)
    image_with_labels = create_image_labels(slic, recon_image, labels, kmeans_labels)
    final_image = combine_images(image_with_labels, palette)
    filename = 'pbk-{}-.png'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
    success = cv2.imwrite(filename, cv2.cvtColor(final_image, cv2.COLOR_RGBA2BGRA))
    return filename, success

def prep_image(image):
   if image.shape[-1] == 3: 
       image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
   _, img = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
   return img.tobytes()

@record_time
def trigger_dalle(image_path, prompt, mode):
    if mode is None:
        return image_path
    client = OpenAI()
    if mode == 'v':
        image = skimage.io.imread(image_path)
        img_bytes = prep_image(image)
        response = client.images.create_variation(
            model="dall-e-2",
            image=img_bytes,
            n=1,
            size='1024x1024'
        )
        return response.data[0].url
    
    if mode == 'C':
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url

