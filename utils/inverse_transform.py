import numpy as np
from numpy.typing import NDArray
from typing import List
from sklearn.compose import ColumnTransformer

def inverse_transform(preprocessor: ColumnTransformer, x: NDArray):
    ohe = preprocessor.transformers_[1][1]
    scaler = preprocessor.transformers_[0][1]

    categories = preprocessor.transformers_[1][1].categories_
    start = preprocessor.transformers_[1][2][0]

    maxs = []
    for i in range(len(categories)):
        arr = x[start:start+len(categories[i])]
        maxs.append(np.argmax(arr))
        start += len(categories[i])

    num_rescaled = scaler.inverse_transform(x[:preprocessor.transformers_[1][2][0]].reshape(1, -1))
    x_rescaled = np.concatenate((num_rescaled[0], np.array(maxs)))

    return x_rescaled