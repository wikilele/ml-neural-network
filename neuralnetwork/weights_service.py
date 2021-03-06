import random

class WeightsService:

    def __init__(self,lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def get_bias(self):
        return self._get_random_value()
    
    def get_weights(self,weights_number):
        weights = []
        for i in range(weights_number):
            weights.append(self._get_random_value())
        return weights

    def _get_random_value(self):
        return random.uniform(self.lower_bound, self.upper_bound)
