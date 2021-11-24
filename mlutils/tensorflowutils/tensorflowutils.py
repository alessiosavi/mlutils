import json
import os

import tensorflow as tf

example_conf = """
{
    "model_type": "Sequential",
    "input_shape": [
        60,
        1
    ],
    "model_conf": [
        {
            "layer": "LSTM",
            "layer_conf": {
                "units": 256,
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
                "use_bias": true,
                "kernel_initializer": "glorot_uniform",
                "recurrent_initializer": "orthogonal",
                "bias_initializer": "zeros",
                "unit_forget_bias": true,
                "dropout": 0.0,
                "recurrent_dropout": 0.0,
                "return_sequences": true,
                "return_state": false,
                "go_backwards": false,
                "stateful": false,
                "time_major": false,
                "unroll": false
            }
        },
        {
            "layer": "GRU",
            "layer_conf": {
                "units": 256,
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
                "use_bias": true,
                "kernel_initializer": "glorot_uniform",
                "recurrent_initializer": "orthogonal",
                "bias_initializer": "zeros",
                "dropout": 0.0,
                "recurrent_dropout": 0.0,
                "return_sequences": false,
                "return_state": false,
                "go_backwards": false,
                "stateful": false,
                "unroll": false,
                "time_major": false,
                "reset_after": true
            }
        },
        {
            "layer": "DROPOUT",
            "layer_conf": {
                "rate": 0.1
            }
        },
        {
            "layer": "DENSE",
            "layer_conf": {
                "units": 64,
                "activation": "",
                "use_bias": true,
                "kernel_initializer": "glorot_uniform",
                "bias_initializer": "zeros"
            }
        }
    ],
    "compile_conf": {
        "optimizer": "adam",
        "loss": "mean_squared_error",
        "metrics": "mean_absolute_percentage_error"
    }
}
"""


def build_model_core(conf: dict) -> tf.keras.Model:
    if isinstance(conf, str):
        conf = json.loads(conf)
    if conf["model_type"] == "Sequential":
        model = tf.keras.Sequential()
    else:
        raise Exception("Model {} not managed!".format(conf["model_type"]))

    for i in range(len(conf["model_conf"])):
        if i == 0:
            input_shape = tuple([None if isinstance(a, str) else a for a in conf["input_shape"]])
            conf["model_conf"][i]["layer_conf"]["input_shape"] = input_shape

        if conf["model_conf"][i]["layer"] == "LSTM":
            model.add(tf.keras.layers.LSTM(**conf["model_conf"][i]["layer_conf"]))

        elif conf["model_conf"][i]["layer"] == "GRU":
            model.add(tf.keras.layers.GRU(**conf["model_conf"][i]["layer_conf"]))

        elif conf["model_conf"][i]["layer"] == "DENSE":
            if "activation" not in conf["model_conf"][i]["layer_conf"] or conf["model_conf"][i]["layer_conf"]["activation"] == "":
                conf["model_conf"][i]["layer_conf"]["activation"] = None
            model.add(tf.keras.layers.Dense(**conf["model_conf"][i]["layer_conf"]))

        elif conf["model_conf"][i]["layer"] == "DROPOUT":
            model.add(tf.keras.layers.Dropout(conf["model_conf"][i]["layer_conf"]["rate"]))

        else:
            raise Exception("Layer {} not managed!".format(conf["model_conf"][i]["layer"]))

    model.compile(**conf["compile_conf"])
    model.summary()
    return model


def fit_model(model, **kwargs) -> tf.keras.Model:
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True
    )
    if "callbacks" in kwargs:
        kwargs["callbacks"].append(callback)
    else:
        print("Overriding callback: {}".format(callback.__dict__))
        kwargs["callbacks"] = [callback]

    if "epochs" not in kwargs:
        kwargs["epochs"] = 1000

    model.fit(**kwargs)
    return model


def train_model(conf, x, y, x_test=None, y_test=None, **kwargs):
    model_arch: str = "model_dir/"
    if not os.path.exists(model_arch):
        os.mkdir(model_arch)
    max_model: str = ""
    for layer_name in conf["model_conf"]:
        model_arch += layer_name["layer"] + "_"
    model_arch = model_arch.removesuffix("_")
    # model_arch = LSTM_GRU_DROPOUT_DENSE, create a folder that will save every model like this one
    if not os.path.exists(model_arch):
        os.mkdir(model_arch)
    _dir = []
    # Creating numerical folder (1, 2, 3, 4) related to the training of the same architecture with different hyperparameters
    for f in os.listdir(model_arch):
        if os.path.isdir(os.path.join(model_arch, f)):
            _dir.append(f)
    _dir = [int(a) for a in _dir if a.isdigit()]
    if len(_dir) == 0:
        max_model = "0"
    else:
        max_model = "{}".format(max(_dir) + 1)

    model = build_model_core(conf)
    model = fit_model(model, x=x, y=y, **kwargs)
    file_path_save = os.path.join(model_arch, max_model)
    model.save(file_path_save)

    if x_test is not None and y_test is not None:
        hist = model.evaluate(x_test, y_test)
        with open(os.path.join(file_path_save, "loss_metric.txt"), "wt") as f:
            f.write("{}".format(hist))
    return model
