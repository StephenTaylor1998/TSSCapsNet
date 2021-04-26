from keras_flops import get_flops
from tensorflow.keras import Model, Input


def compute(model, input_shape, batch_size=1):
    inp = Input(input_shape)
    out = model(inp)
    _model = Model(inp, out)
    flops = get_flops(_model, batch_size=batch_size)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")

