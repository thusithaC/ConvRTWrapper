import tensorflow_hub as tfhub
import tensorflow as tf
import requests
import os.path
import tarfile
from sklearn.decomposition import PCA

import logging


logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("ConvRTModel")

import tensorflow_text  # required for tokenization ops

TF_MODEL = "http://models.poly-ai.com/convert/v1/model.tar.gz"
TF_MODEL_FILE = "/tmp/poly-ai-model-convert-v1.tar.gz"
TF_MODEL_UNZIP_LOCATION = "/tmp/poly-ai-model-convert-v1/"


class ConvRTModel:
    def __init__(self, dim_reduction=False, n_components=128):
        '''
        :param dim_reduction: Carry out PCA based dimensionality reduction for the embeddings. default is False
        :param n_components: Only valid if dim_reduction==True. The number of output coefficients
        '''
        self.model = self.load()
        self.dim_reduction = dim_reduction
        if dim_reduction:
            self.n_components = n_components
        else:
            self.n_components = 1024  # constant for now


    def load(self):
        if os.path.isfile(TF_MODEL_FILE):
            logger.info("File exists locally")
        else:
            logger.info("Downloading model file")
            resource = requests.get(TF_MODEL)
            with open(TF_MODEL_FILE, 'wb') as outfile:
                outfile.write(resource.content)

        tar = tarfile.open(TF_MODEL_FILE)
        tar.extractall(TF_MODEL_UNZIP_LOCATION)
        module = tfhub.load(TF_MODEL_UNZIP_LOCATION)
        return module

    def _do_pca(self, X):
        pca = PCA(n_components=self.n_components)
        return pca.fit_transform(X)

    def get_embeddings(self, sentences: list):
        '''
        :param sentences: List of sentences (str instances)
        :return: list of embeddings
        '''
        x = tf.constant(sentences)
        y = self.model.signatures['default'](x)['default'].numpy()
        if self.dim_reduction:
            return self._do_pca(y)
        else:
            return y

model = ConvRTModel()