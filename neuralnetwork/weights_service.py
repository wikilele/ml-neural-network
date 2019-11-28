import random

class WeightsService:

    def __init__(lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def get_bias():
        return get_random_value()
    
    def get_weights(weights_number):
        weights = []
        for i in range(weights_number):
            weights.append(get_random_value())
        return weights

    def get_random_value():
        return random.uniform(self.lower_bound, self.upper_bound)
