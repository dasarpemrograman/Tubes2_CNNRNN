from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import numpy as np


class ScratchLayer(Protocol):
    def forward(self, x: np.ndarray) -> np.ndarray:
        ...


@dataclass
class Conv2D:
    kernel: np.ndarray
    bias: np.ndarray
    stride: tuple[int, int] = (1, 1)
    padding: str = "valid"

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x is NHWC: batch, height, width, channels.
        if x.ndim != 4:
            raise ValueError(f"Conv2D expects input shape (N, H, W, C), got {x.shape}.")
        kernel = np.asarray(self.kernel, dtype=np.float32)
        bias = np.asarray(self.bias, dtype=np.float32)
        if kernel.ndim != 4:
            raise ValueError("Conv2D kernel must have shape (kH, kW, C_in, C_out).")
        if bias.shape != (kernel.shape[-1],):
            raise ValueError("Conv2D bias must have shape (C_out,).")
        if x.shape[-1] != kernel.shape[2]:
            raise ValueError(f"Input channels {x.shape[-1]} do not match kernel channels {kernel.shape[2]}.")

        x_padded = _pad_input(x, kernel.shape[:2], self.stride, self.padding)
        out_h, out_w = _conv_output_shape(x_padded.shape[1:3], kernel.shape[:2], self.stride)
        output = np.empty((x.shape[0], out_h, out_w, kernel.shape[-1]), dtype=np.float32)

        for row in range(out_h):
            row_start = row * self.stride[0]
            row_end = row_start + kernel.shape[0]
            for col in range(out_w):
                col_start = col * self.stride[1]
                col_end = col_start + kernel.shape[1]
                patch = x_padded[:, row_start:row_end, col_start:col_end, :]
                output[:, row, col, :] = np.tensordot(
                    patch,
                    kernel,
                    axes=((1, 2, 3), (0, 1, 2)),
                ) + bias

        return output


@dataclass
class LocallyConnected2D:
    kernel: np.ndarray
    bias: np.ndarray | None = None
    kernel_size: tuple[int, int] | None = None
    stride: tuple[int, int] = (1, 1)
    padding: str = "valid"

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Each spatial output position owns a different kernel, so weights are
        # not shared across rows/columns as they are in Conv2D.
        if x.ndim != 4:
            raise ValueError(
                f"LocallyConnected2D expects input shape (N, H, W, C), got {x.shape}."
            )

        kernel = np.asarray(self.kernel, dtype=np.float32)
        bias = None if self.bias is None else np.asarray(self.bias, dtype=np.float32)
        k_h, k_w, out_h, out_w, out_channels = self._infer_shapes(x, kernel)
        x_padded = _pad_input(x, (k_h, k_w), self.stride, self.padding)
        output = np.empty((x.shape[0], out_h, out_w, out_channels), dtype=np.float32)

        for row in range(out_h):
            row_start = row * self.stride[0]
            row_end = row_start + k_h
            for col in range(out_w):
                col_start = col * self.stride[1]
                col_end = col_start + k_w
                patch = x_padded[:, row_start:row_end, col_start:col_end, :]
                local_kernel = _local_kernel_at(kernel, row, col, out_h, out_w)
                patch_flat = patch.reshape(x.shape[0], -1)
                kernel_flat = local_kernel.reshape(-1, out_channels)
                output[:, row, col, :] = patch_flat @ kernel_flat

        if bias is not None:
            output += _reshape_local_bias(bias, out_h, out_w, out_channels)

        return output

    def _infer_shapes(self, x: np.ndarray, kernel: np.ndarray) -> tuple[int, int, int, int, int]:
        if kernel.ndim == 6:
            out_h, out_w, k_h, k_w, in_channels, out_channels = kernel.shape
        elif kernel.ndim == 3:
            out_positions, flat_patch, out_channels = kernel.shape
            k_h, k_w = self._kernel_size_from_flat_patch(flat_patch, x.shape[-1])
            out_h, out_w = _conv_output_shape(
                _pad_input(x, (k_h, k_w), self.stride, self.padding).shape[1:3],
                (k_h, k_w),
                self.stride,
            )
            if out_positions != out_h * out_w:
                raise ValueError(
                    "LocallyConnected2D flattened kernel first dimension must equal out_h * out_w."
                )
            in_channels = x.shape[-1]
        elif kernel.ndim == 4:
            out_positions, flat_patch, in_channels, out_channels = kernel.shape
            if flat_patch != 1:
                raise ValueError("Unsupported 4D local kernel shape.")
            k_h, k_w = self._kernel_size_from_flat_patch(flat_patch * in_channels, x.shape[-1])
            out_h, out_w = _conv_output_shape(
                _pad_input(x, (k_h, k_w), self.stride, self.padding).shape[1:3],
                (k_h, k_w),
                self.stride,
            )
            if out_positions != out_h * out_w:
                raise ValueError(
                    "LocallyConnected2D flattened kernel first dimension must equal out_h * out_w."
                )
        else:
            raise ValueError(
                "LocallyConnected2D kernel must be shape "
                "(out_h, out_w, kH, kW, C_in, C_out) or "
                "(out_h*out_w, kH*kW*C_in, C_out)."
            )

        if in_channels != x.shape[-1]:
            raise ValueError(f"Input channels {x.shape[-1]} do not match local kernel channels {in_channels}.")
        return k_h, k_w, out_h, out_w, out_channels

    def _kernel_size_from_flat_patch(self, flat_patch: int, in_channels: int) -> tuple[int, int]:
        if self.kernel_size is None:
            raise ValueError("kernel_size is required for flattened LocallyConnected2D kernels.")
        k_h, k_w = self.kernel_size
        if flat_patch != k_h * k_w * in_channels:
            raise ValueError("Flattened local kernel does not match kernel_size and input channels.")
        return k_h, k_w


@dataclass
class MaxPooling2D:
    pool_size: tuple[int, int] = (2, 2)
    stride: tuple[int, int] | None = None
    padding: str = "valid"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return _pool2d(
            x,
            self.pool_size,
            self.stride or self.pool_size,
            self.padding,
            np.max,
            pad_value=-np.inf,
        )


@dataclass
class AveragePooling2D:
    pool_size: tuple[int, int] = (2, 2)
    stride: tuple[int, int] | None = None
    padding: str = "valid"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return _pool2d(
            x,
            self.pool_size,
            self.stride or self.pool_size,
            self.padding,
            np.mean,
            pad_value=0.0,
        )


class GlobalAveragePooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"GlobalAveragePooling2D expects (N, H, W, C), got {x.shape}.")
        return x.mean(axis=(1, 2))


class GlobalMaxPooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"GlobalMaxPooling2D expects (N, H, W, C), got {x.shape}.")
        return x.max(axis=(1, 2))


class Flatten:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.reshape((x.shape[0], -1))


@dataclass
class Dense:
    weights: np.ndarray
    bias: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"Dense expects input shape (N, features), got {x.shape}.")
        weights = np.asarray(self.weights, dtype=np.float32)
        bias = np.asarray(self.bias, dtype=np.float32)
        if weights.ndim != 2:
            raise ValueError("Dense weights must have shape (features, units).")
        if x.shape[-1] != weights.shape[0]:
            raise ValueError(f"Input features {x.shape[-1]} do not match Dense weights {weights.shape[0]}.")
        return x @ weights + bias


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)


class Softmax:
    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)


@dataclass
class ScratchSequential:
    layers: list[ScratchLayer]

    def forward(self, x: np.ndarray) -> np.ndarray:
        output = np.asarray(x, dtype=np.float32)
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


ScratchCNNModel = ScratchSequential


def scratch_model_from_keras_model(keras_model: Any) -> ScratchSequential:
    scratch_layers: list[ScratchLayer] = []

    for keras_layer in keras_model.layers:
        class_name = keras_layer.__class__.__name__
        if class_name in {"InputLayer", "Dropout"}:
            continue
        if class_name == "Conv2D":
            scratch_layers.extend(_convert_conv2d(keras_layer))
        elif class_name == "LocallyConnected2D":
            scratch_layers.extend(_convert_locally_connected2d(keras_layer))
        elif class_name == "MaxPooling2D":
            scratch_layers.append(
                MaxPooling2D(
                    pool_size=_as_tuple(keras_layer.pool_size),
                    stride=_as_tuple(keras_layer.strides or keras_layer.pool_size),
                    padding=keras_layer.padding,
                )
            )
        elif class_name == "AveragePooling2D":
            scratch_layers.append(
                AveragePooling2D(
                    pool_size=_as_tuple(keras_layer.pool_size),
                    stride=_as_tuple(keras_layer.strides or keras_layer.pool_size),
                    padding=keras_layer.padding,
                )
            )
        elif class_name == "GlobalAveragePooling2D":
            scratch_layers.append(GlobalAveragePooling2D())
        elif class_name == "GlobalMaxPooling2D":
            scratch_layers.append(GlobalMaxPooling2D())
        elif class_name == "Flatten":
            scratch_layers.append(Flatten())
        elif class_name == "Dense":
            scratch_layers.extend(_convert_dense(keras_layer))
        elif class_name == "Activation":
            scratch_layers.append(_activation_from_name(keras_layer.activation.__name__))
        else:
            raise ValueError(f"Unsupported Keras layer for scratch conversion: {class_name}")

    return ScratchSequential(scratch_layers)


def _convert_conv2d(keras_layer: Any) -> list[ScratchLayer]:
    weights = keras_layer.get_weights()
    if len(weights) == 1:
        kernel = weights[0]
        bias = np.zeros((kernel.shape[-1],), dtype=np.float32)
    else:
        kernel, bias = weights
    layers: list[ScratchLayer] = [
        Conv2D(
            kernel=np.asarray(kernel, dtype=np.float32),
            bias=np.asarray(bias, dtype=np.float32),
            stride=_as_tuple(keras_layer.strides),
            padding=keras_layer.padding,
        )
    ]
    layers.extend(_activation_layers(keras_layer.activation.__name__))
    return layers


def _convert_locally_connected2d(keras_layer: Any) -> list[ScratchLayer]:
    weights = keras_layer.get_weights()
    if len(weights) == 1:
        kernel = weights[0]
        bias = None
    else:
        kernel, bias = weights
    layers: list[ScratchLayer] = [
        LocallyConnected2D(
            kernel=np.asarray(kernel, dtype=np.float32),
            bias=None if bias is None else np.asarray(bias, dtype=np.float32),
            kernel_size=_as_tuple(keras_layer.kernel_size),
            stride=_as_tuple(keras_layer.strides),
            padding=keras_layer.padding,
        )
    ]
    layers.extend(_activation_layers(keras_layer.activation.__name__))
    return layers


def _convert_dense(keras_layer: Any) -> list[ScratchLayer]:
    weights, bias = keras_layer.get_weights()
    layers: list[ScratchLayer] = [
        Dense(
            weights=np.asarray(weights, dtype=np.float32),
            bias=np.asarray(bias, dtype=np.float32),
        )
    ]
    layers.extend(_activation_layers(keras_layer.activation.__name__))
    return layers


def _activation_layers(name: str) -> list[ScratchLayer]:
    if name in {"linear", "identity"}:
        return []
    return [_activation_from_name(name)]


def _activation_from_name(name: str) -> ScratchLayer:
    if name == "relu":
        return ReLU()
    if name == "softmax":
        return Softmax()
    raise ValueError(f"Unsupported activation for scratch CNN: {name}")


def _pool2d(
    x: np.ndarray,
    pool_size: tuple[int, int],
    stride: tuple[int, int],
    padding: str,
    reducer: Any,
    pad_value: float,
) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"Pooling expects input shape (N, H, W, C), got {x.shape}.")
    x_padded = _pad_input(x, pool_size, stride, padding, constant_values=pad_value)
    out_h, out_w = _conv_output_shape(x_padded.shape[1:3], pool_size, stride)
    output = np.empty((x.shape[0], out_h, out_w, x.shape[-1]), dtype=np.float32)

    for row in range(out_h):
        row_start = row * stride[0]
        row_end = row_start + pool_size[0]
        for col in range(out_w):
            col_start = col * stride[1]
            col_end = col_start + pool_size[1]
            patch = x_padded[:, row_start:row_end, col_start:col_end, :]
            output[:, row, col, :] = reducer(patch, axis=(1, 2))

    return output


def _pad_input(
    x: np.ndarray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: str,
    constant_values: float = 0.0,
) -> np.ndarray:
    if padding == "valid":
        return x
    if padding != "same":
        raise ValueError(f"Unsupported padding: {padding}")

    in_h, in_w = x.shape[1:3]
    out_h = int(np.ceil(in_h / stride[0]))
    out_w = int(np.ceil(in_w / stride[1]))
    pad_h = max((out_h - 1) * stride[0] + kernel_size[0] - in_h, 0)
    pad_w = max((out_w - 1) * stride[1] + kernel_size[1] - in_w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return np.pad(
        x,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=constant_values,
    )


def _conv_output_shape(
    input_hw: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
) -> tuple[int, int]:
    out_h = (input_hw[0] - kernel_size[0]) // stride[0] + 1
    out_w = (input_hw[1] - kernel_size[1]) // stride[1] + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Kernel/pool size is larger than the padded input.")
    return out_h, out_w


def _local_kernel_at(
    kernel: np.ndarray,
    row: int,
    col: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    if kernel.ndim == 6:
        return kernel[row, col]
    if kernel.ndim == 3:
        return kernel[row * out_w + col]
    raise ValueError(f"Unsupported local kernel shape: {kernel.shape}")


def _reshape_local_bias(
    bias: np.ndarray,
    out_h: int,
    out_w: int,
    out_channels: int,
) -> np.ndarray:
    if bias.shape == (out_channels,):
        return bias.reshape(1, 1, 1, out_channels)
    if bias.shape == (out_h, out_w, out_channels):
        return bias.reshape(1, out_h, out_w, out_channels)
    if bias.shape == (out_h * out_w, out_channels):
        return bias.reshape(1, out_h, out_w, out_channels)
    raise ValueError(f"Unsupported local bias shape: {bias.shape}")


def _as_tuple(value: Iterable[int] | int) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    items = tuple(int(item) for item in value)
    if len(items) != 2:
        raise ValueError(f"Expected a 2D tuple, got {value}.")
    return items
