from multiprocessing import Process
from typing import Mapping, Any
from succesive_halving import SuccessiveHalving


class Worker(Process):
    def __init__(self, kwargs: Mapping[str, Any] = ...) -> None:
        super().__init__(kwargs=kwargs, group=None)
    
    def run(self):
        #print(self._kwargs)
        sh = SuccessiveHalving(**self._kwargs)
        scores, configurations, candidates = sh.run()
        return scores, configurations, candidates
