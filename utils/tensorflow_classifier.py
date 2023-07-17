import numpy as np
import tensorflow as tf

class TensorflowClassifier:
    def __init__(self, classifier):
        self.classifier = classifier
    
    def predict_proba(self, x):
        print(f'classifier {self.classifier}')
        infer = self.classifier.signatures['serving_default']
        print(f'infer {infer}')
        return np.array(infer(tf.convert_to_tensor(x, dtype=tf.float32))['dense_3'])
    
    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

class LcldTensorflowClassifier:
    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, x, y):
        self.classifier.fit(x, y)

    def predict_proba(self, x):
        return self.classifier.predict(x)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def get_internal_classifier(self):
        return self.classifier