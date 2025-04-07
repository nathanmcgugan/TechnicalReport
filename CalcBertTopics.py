import pickle

from sentence_transformers import SentenceTransformer

from bertopic import BERTopic
import numpy as np
import pandas as pd

topic_model = BERTopic.load('TopicModel')

with open('combined_training.pkl', 'rb') as handle:
    texts = pickle.load(handle)

topics = np.array(topic_model.transform(texts)[1])
np.save('topics.npy', topics)
