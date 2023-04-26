import numpy as np
import tensorflow as tf

class TensorflowClassifier:
    def __init__(self, classifier):
        self.classifier = classifier
    
    def predict_proba(self, x):
        infer = self.classifier.signatures['serving_default']
        return np.array(infer(tf.convert_to_tensor(x, dtype=tf.float32))['dense_2'])
    
    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)