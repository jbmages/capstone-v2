import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score



