
class Regularization:

    def compute_regularization(self):
        # no regularization
        return 0

class TikhonovRegularization(Regularization):

    def __init__(self, regularization_lambda):
        self.regularization_lambda = regularization_lambda

    def compute_regularization(self):
        pass
