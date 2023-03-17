import numpy as np
from scipy.special import softmax
from typing import List, Any
from numpy.typing import NDArray
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class SuccessRateCalculator(ABC):
    classifier: Any
    data: Any
    labels: Any
    scores: Any
    candidates: List

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError()
    

class TfCalculator(SuccessRateCalculator):
    def __init__(self, classifier, data, labels, scores, candidates):
        super().__init__(classifier=classifier, data=data, labels=labels, scores=scores, candidates=candidates)

    def evaluate(self):
        correct = 0
        success_rate = 0
        adversarials = []
        best_candidates = []
        for i, (x, y) in enumerate(zip(self.data, self.labels)):
            pred = np.argmax(softmax(self.classifier.predict(x[np.newaxis, :])))
            if pred != y:
                #print('inside the if')
                continue

            correct += 1
            best_score_idx = np.argmin(self.scores[i])  
            best_candidate = self.candidates[i][best_score_idx]
            pred = np.argmax(softmax(self.classifier.predict(best_candidate[np.newaxis, :])))
            best_candidates.append(best_candidate)

            if pred != y:
                #print(f'adversarial {i}')
                adversarials.append(best_candidate)
                success_rate += 1
        eps = 0.0001 if correct == 0 else 0
        
        return round(success_rate / correct + eps, 3), best_candidates, adversarials


