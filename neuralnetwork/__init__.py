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