from .model_builder import ModelBuilder


mb = ModelBuilder()

def input_layer(units):
    return mb.input_layer(units)

def hidden_layer(units,**kwargs):
    return mb.hidden_layer(units,**kwargs)

def output_layer(units,**kwargs):
    return mb.output_layer(units,**kwargs)

def build():
    model = mb.build()
    reset()
    return model

def reset():
    global mb
    mb = ModelBuilder()

def init_weights_random(bound):
    return mb.init_weights_random(bound)

def learning_rate( lr, tau_decay):
    return mb.learning_rate( lr, tau_decay)

def momentum(alpha, use_nesterov):
    return mb.momentum(alpha, use_nesterov)

def regularization(lambda_param):
    return mb.regularization(lambda_param)


def from_parameters(params, hidden_activation_func, output_activation_func):
    ''' init the model using a json/dict with the parameters'''
    weights_bound = params['weights_bound']
    l_rate = params['learning_rate']
    momentum_alpha = params['momentum_alpha']
    regularization_lambda = params['regularization_lambda']

    input_dim = params['input_layer_dim']
    hidden_layer_num = params['hidden_layer_number']
    hidden_layer_dim = params['hidden_layer_dim'] # should be a list of x elements where x is the hidden_layer_num
    output_layer_dim = params['output_layer_dim']

    input_layer(input_dim)
    for i in range(hidden_layer_num):
        hidden_layer(hidden_layer_dim[i], activation=hidden_activation_func)
    output_layer(output_layer_dim, activation=output_activation_func)
            
    init_weights_random(weights_bound)
    learning_rate(l_rate,0)
    momentum(momentum_alpha, use_nesterov=False)
    regularization(regularization_lambda)