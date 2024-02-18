import numpy as np
import sklearn.neighbors
import scipy.sparse
import cv2
import time
from scipy.sparse.linalg import cg, spsolve, lsqr

def knn_matting(image, trimap, my_lambda=100):
    [h, w, c] = image.shape
    image = image.astype(np.float32) / 255.0
    trimap = trimap / 255.0
    foreground = (trimap == 1.0).astype(int)
    background = (trimap == 0.0).astype(int)
    
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0] 
    s_channel = hsv[:, :, 1] 
    v_channel = hsv[:, :, 2] 
    
    
    cos_h = np.cos(h_channel * 2 * np.pi)
    sin_h = np.sin(h_channel * 2 * np.pi)
    x_coord = np.repeat(np.arange(h)[:, np.newaxis], w, axis=1)
    y_coord = np.repeat(np.arange(w)[np.newaxis, :], h, axis=0)
    
    feature_vector = np.stack([cos_h, sin_h, s_channel, v_channel, x_coord, y_coord], axis=-1)
    feature_vector = feature_vector.reshape(-1, 6)
    
    
    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=10, n_jobs=4).fit(feature_vector)
    distances, indices = knn.kneighbors(feature_vector)
    
   
    C = np.max(distances)
    weights = 1 - distances / C
    rows = np.repeat(np.arange(h * w), 10)
    A = scipy.sparse.coo_matrix((weights.ravel(), (rows, indices.ravel())), shape=(h * w, h * w))
    
    
    D = scipy.sparse.diags(np.ravel(A.sum(axis=1)))
    L = D - A
    
   
    marked = (foreground + background).astype(float).ravel()
    M = scipy.sparse.diags(marked)
    v = foreground.ravel().astype(float)
    
   
    alpha= cg(L + my_lambda * M, my_lambda * v)[0]
    alpha = alpha.reshape(h, w)
    
    return alpha

if __name__ == '__main__':
    start_time = time.time()
    image = cv2.imread('./image/bear.png')
    trimap = cv2.imread('./trimap/bear.png', cv2.IMREAD_GRAYSCALE)

    alpha = knn_matting(image, trimap)
    alpha = alpha[:, :, np.newaxis]
   
    background = cv2.imread('sky.png')   
    background = cv2.resize(background, (image.shape[1], image.shape[0]))
    result =   alpha * image   + (1 - alpha) * background 

    
    cv2.imwrite('./result/bear.png', result)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"執行時間：{execution_time} 秒")



