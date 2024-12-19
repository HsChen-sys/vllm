"""Utilities for selecting and loading models."""
import contextlib
from typing import Tuple, Type

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.model_executor.models import ModelRegistry

#只需要在一个函数上加上 @contextlib.contextmanager 装饰器，然后在函数中用 yield 将控制权分段即可。你不必显式编写 __enter__ 和 __exit__。
@contextlib.contextmanager #上下文管理装饰器
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()  # 记住当前默认dtype
    torch.set_default_dtype(dtype) # 设置新的默认dtype
    yield # 暂停，进入 with 代码块
    torch.set_default_dtype(old_dtype) # 退出时恢复旧dtype


def get_model_architecture(
        model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])
    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    if (model_config.quantization is not None
            and model_config.quantization != "fp8"
            and "MixtralForCausalLM" in architectures):
        architectures = ["QuantMixtralForCausalLM"]

    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return (model_cls, arch)
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]
