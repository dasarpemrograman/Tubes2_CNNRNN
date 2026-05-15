from __future__ import annotations

import keras
from keras import layers
import tensorflow as tf

try:
    from cnn.configs import CNNModelConfig, DEFAULT_IMAGE_SIZE, INTEL_CLASS_NAMES
except ModuleNotFoundError:
    from configs import CNNModelConfig, DEFAULT_IMAGE_SIZE, INTEL_CLASS_NAMES


@keras.saving.register_keras_serializable(package="cnn")
class LocallyConnected2D(layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int | tuple[int, int],
        strides: int | tuple[int, int] = (1, 1),
        padding: str = "same",
        activation: str | None = None,
        use_bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = _as_tuple(kernel_size)
        self.strides = _as_tuple(strides)
        self.padding = padding
        self.activation = keras.activations.get(activation)
        self.activation_name = keras.activations.serialize(self.activation)
        self.use_bias = use_bias

    def build(self, input_shape) -> None:
        input_height, input_width, input_channels = input_shape[1], input_shape[2], input_shape[3]
        if input_height is None or input_width is None or input_channels is None:
            raise ValueError("LocallyConnected2D requires known input height, width, and channels.")

        if self.padding == "same":
            out_height = int((input_height + self.strides[0] - 1) // self.strides[0])
            out_width = int((input_width + self.strides[1] - 1) // self.strides[1])
        elif self.padding == "valid":
            out_height = (input_height - self.kernel_size[0]) // self.strides[0] + 1
            out_width = (input_width - self.kernel_size[1]) // self.strides[1] + 1
        else:
            raise ValueError("LocallyConnected2D padding must be 'same' or 'valid'.")

        self.output_spatial_shape = (out_height, out_width)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(
                out_height,
                out_width,
                self.kernel_size[0],
                self.kernel_size[1],
                int(input_channels),
                self.filters,
            ),
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(out_height, out_width, self.filters),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs):
        padded = self._pad_input(inputs)
        outputs = []
        out_height, out_width = self.output_spatial_shape
        for row in range(out_height):
            row_start = row * self.strides[0]
            row_end = row_start + self.kernel_size[0]
            row_outputs = []
            for col in range(out_width):
                col_start = col * self.strides[1]
                col_end = col_start + self.kernel_size[1]
                patch = padded[:, row_start:row_end, col_start:col_end, :]
                local_kernel = self.kernel[row, col]
                value = tf.tensordot(patch, local_kernel, axes=[[1, 2, 3], [0, 1, 2]])
                if self.bias is not None:
                    value = value + self.bias[row, col]
                row_outputs.append(value)
            outputs.append(tf.stack(row_outputs, axis=1))
        output = tf.stack(outputs, axis=1)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.output_spatial_shape, self.filters)

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "activation": self.activation_name,
                "use_bias": self.use_bias,
            }
        )
        return config

    def _pad_input(self, inputs):
        if self.padding == "valid":
            return inputs
        input_shape = tf.shape(inputs)
        in_height = input_shape[1]
        in_width = input_shape[2]
        out_height, out_width = self.output_spatial_shape
        pad_height = max((out_height - 1) * self.strides[0] + self.kernel_size[0] - int(inputs.shape[1]), 0)
        pad_width = max((out_width - 1) * self.strides[1] + self.kernel_size[1] - int(inputs.shape[2]), 0)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        del in_height, in_width
        return tf.pad(inputs, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])


def build_keras_cnn_model(
    config: CNNModelConfig,
    input_shape: tuple[int, int, int] = (*DEFAULT_IMAGE_SIZE, 3),
    num_classes: int = len(INTEL_CLASS_NAMES),
    shared_parameters: bool = True,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs

    for index in range(config.conv_layers):
        if shared_parameters:
            x = layers.Conv2D(
                filters=config.filters[index],
                kernel_size=config.kernel_sizes[index],
                padding="same",
                activation="relu",
                name=f"conv2d_{index + 1}",
            )(x)
        else:
            x = LocallyConnected2D(
                filters=config.filters[index],
                kernel_size=config.kernel_sizes[index],
                padding="same",
                activation="relu",
                name=f"locally_connected2d_{index + 1}",
            )(x)
        if config.pooling_type == "max":
            x = layers.MaxPooling2D(pool_size=(2, 2), name=f"max_pool_{index + 1}")(x)
        else:
            x = layers.AveragePooling2D(pool_size=(2, 2), name=f"avg_pool_{index + 1}")(x)

    if config.classifier_pooling == "global_average":
        x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    elif config.classifier_pooling == "global_max":
        x = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    else:
        x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(config.dense_units, activation="relu", name="dense_classifier")(x)
    if config.dropout_rate > 0:
        x = layers.Dropout(config.dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=config.model_id)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_keras_locally_connected_cnn_model(
    config: CNNModelConfig,
    input_shape: tuple[int, int, int] = (*DEFAULT_IMAGE_SIZE, 3),
    num_classes: int = len(INTEL_CLASS_NAMES),
) -> keras.Model:
    return build_keras_cnn_model(
        config,
        input_shape=input_shape,
        num_classes=num_classes,
        shared_parameters=False,
    )


def _as_tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    return (int(value[0]), int(value[1]))
