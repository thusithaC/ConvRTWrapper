import tensorflow_hub as tfhub
import tensorflow as tf
import requests
import os.path
import gzip
import tarfile
import shutil
import logging

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("ConvRTModel")

import tensorflow_text  # required for tokenization ops

TF_MODEL = "http://models.poly-ai.com/convert/v1/model.tar.gz"
TF_MODEL_FILE = "/tmp/poly-ai-model-convert-v1.tar.gz"
TF_MODEL_UNZIP_LOCATION = "/tmp/poly-ai-model-convert-v1/"

# resource = requests.get(TF_MODEL)
#
# with open(TF_MODEL_FILE, 'wb') as outfile:
#     outfile.write(resource.content)
#
# module = tfhub.load()
#
#
# x = tf.constant(["hello how are you?", "what is your name?", "thank you good bye"])
# sentence_encodings = module.signatures['default'](x)
#
# print(sentence_encodings)


class ConvRTModel:
    def __init__(self):
        self.model = self.load()

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

    def get_embeddings(self, sentences: list):
        '''
        :param sentences: List of sentences (str instances)
        :return: list of embeddings
        '''
        x = tf.constant(sentences)
        y = self.model.signatures['default'](x)
        return y['default'].numpy()

model = ConvRTModel()