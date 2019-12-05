from .model_builder import ModelBuilder


mb = ModelBuilder()

def input_layer(units):
    return mb.input_layer(units)

def hidden_layer(units,**kwargs):
    return mb.hidden_layer(units,**kwargs)

def output_layer(units,**kwargs):
    return mb.output_layer(units,**kwargs)

def build():
    return mb.build()

def weights_service(ws):
    return mb.weights_service(ws)