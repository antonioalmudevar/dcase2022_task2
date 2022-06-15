from typing import Callable, Any, Optional, List

import torch
from torch import Tensor
from torch import nn

__all__ = ["mobilenetv2", "mobilenetv2_reduced"]


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvNormActivation(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU6,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels


class InvertedResidual(nn.Module):
    def __init__(
        self, 
        inp: int, 
        oup: int, 
        stride: int, 
        expand_ratio: int, 
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                ConvNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        input_channel: int,
        last_channel: int,
        inverted_residual_setting: Optional[List[List[int]]],
        embedding_dim: int,
        width_mult: float = 1.0,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
    
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            ConvNormActivation(1, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(
            ConvNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # getting embeddings
        self.embeddings = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, embedding_dim),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.embeddings(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _mobilenet_v2(input_channel, last_channel, inverted_residual_setting, **kwargs: Any) -> MobileNetV2:
    model = MobileNetV2(input_channel, last_channel, inverted_residual_setting, **kwargs)
    return model


def mobilenetv2(**kwargs: Any) -> MobileNetV2:
    input_channel = 32
    last_channel = 1280
    inverted_residual_setting = [
        # t, c, n, s
        [1,  16, 1, 1],
        [6,  24, 2, 2],
        [6,  32, 3, 2],
        [6,  64, 4, 2],
        [6,  96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    model = _mobilenet_v2(input_channel, last_channel, inverted_residual_setting, **kwargs)
    return model


def mobilenetv2_reduced(**kwargs: Any) -> MobileNetV2:
    input_channel = 16
    last_channel = 640
    inverted_residual_setting = [
        # t, c, n, s
        [1,   8, 1, 1],
        [6,  12, 2, 2],
        [6,  16, 3, 2],
        [6,  32, 4, 2],
        [6,  48, 3, 1],
        [6,  80, 3, 2],
        [6, 160, 1, 1],
    ]
    model = _mobilenet_v2(input_channel, last_channel, inverted_residual_setting, **kwargs)
    return model

def mobilenetv2_extended(**kwargs: Any) -> MobileNetV2:
    input_channel = 64
    last_channel = 2560
    inverted_residual_setting = [
        # t, c, n, s
        [1,  32, 1, 1],
        [6,  48, 2, 2],
        [6,  64, 3, 2],
        [6, 128, 4, 2],
        [6, 192, 3, 1],
        [6, 320, 3, 2],
        [6, 640, 1, 1],
    ]
    model = _mobilenet_v2(input_channel, last_channel, inverted_residual_setting, **kwargs)
    return model