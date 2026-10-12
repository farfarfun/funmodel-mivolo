"""timm 改编实现。

Copyright 2023 Irina Tolstykh, Maxim Kuprashevich
Copyright 2020 Ross Wightman

Licensed under the Apache License, Version 2.0.  See LICENSE-APACHE-2.0.
Derived from https://github.com/huggingface/pytorch-image-models and modified for MiVOLO.
"""

import os
from typing import Any

import timm



# register new models
from funmodel.mivolo.model.mivolo_model import *  # noqa: F403, F401
from timm.layers import set_layer_config
from timm.models._factory import parse_model_name
from timm.models._helpers import load_state_dict, remap_checkpoint
from timm.models._hub import load_model_config_from_hf
from timm.models._pretrained import PretrainedCfg, split_model_name_tag
from timm.models._registry import is_model, model_entrypoint


def load_checkpoint(
    model: Any,
    checkpoint_path: str,
    use_ema: bool = True,
    strict: bool = True,
    remap: bool = False,
    filter_keys: list[str] | None = None,
    state_dict_map: dict[str, str] | None = None,
) -> Any:
    """将检查点加载到 timm 模型中。

    Args:
        model: 待加载权重的模型实例。
        checkpoint_path: 检查点文件路径。
        use_ema: 是否优先加载 EMA 权重。
        strict: 是否严格匹配状态字典键。
        remap: 是否使用 timm 重映射检查点。
        filter_keys: 要从状态字典滤除的键名片段。
        state_dict_map: 状态字典键名替换映射。

    Returns:
        PyTorch 返回的不兼容键集合；加载 numpy 权重时返回 ``None``。
    """
    if os.path.splitext(checkpoint_path)[-1].lower() in (".npz", ".npy"):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, "load_pretrained"):
            timm.models._model_builder.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError("Model cannot load numpy checkpoint")
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
    if remap:
        state_dict = remap_checkpoint(model, state_dict)
    if filter_keys:
        for sd_key in list(state_dict.keys()):
            for filter_key in filter_keys:
                if filter_key in sd_key:
                    if sd_key in state_dict:
                        del state_dict[sd_key]

    rep = []
    if state_dict_map is not None:
        # 'patch_embed.conv1.' : 'patch_embed.conv.'
        for state_k in list(state_dict.keys()):
            for target_k, target_v in state_dict_map.items():
                if target_v in state_k:
                    target_name = state_k.replace(target_v, target_k)
                    state_dict[target_name] = state_dict[state_k]
                    rep.append(state_k)
        for r in rep:
            if r in state_dict:
                del state_dict[r]

    incompatible_keys = model.load_state_dict(state_dict, strict=strict if filter_keys is None else False)
    return incompatible_keys


def create_model(
    model_name: str,
    pretrained: bool = False,
    pretrained_cfg: str | dict[str, Any] | PretrainedCfg | None = None,
    pretrained_cfg_overlay: dict[str, Any] | None = None,
    checkpoint_path: str = "",
    scriptable: bool | None = None,
    exportable: bool | None = None,
    no_jit: bool | None = None,
    filter_keys: list[str] | None = None,
    state_dict_map: dict[str, str] | None = None,
    **kwargs: Any,
) -> Any:
    """根据 timm 注册名称创建模型。

    Args:
        model_name: 模型名称，可包含 Hugging Face Hub 来源。
        pretrained: 是否加载预训练权重。
        pretrained_cfg: 预训练配置或配置标签。
        pretrained_cfg_overlay: 覆盖预训练配置的字段。
        checkpoint_path: 创建后额外加载的检查点路径。
        scriptable: 是否启用 TorchScript 兼容配置。
        exportable: 是否启用导出兼容配置。
        no_jit: 是否禁用 JIT 层。
        filter_keys: 加载检查点时要滤除的键名片段。
        state_dict_map: 加载检查点时的键名替换映射。
        **kwargs: 传递给模型构造函数的额外参数。

    Returns:
        创建完成的模型实例。
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_name = parse_model_name(model_name)
    if model_source == "hf-hub":
        if pretrained_cfg:
            raise ValueError("从 Hugging Face Hub 创建模型时不能设置 pretrained_cfg")
        # For model names specified in the form `hf-hub:path/architecture_name@revision`,
        # load model weights + pretrained_cfg from Hugging Face hub.
        pretrained_cfg, model_name = load_model_config_from_hf(model_name)
    else:
        model_name, pretrained_tag = split_model_name_tag(model_name)
        if not pretrained_cfg:
            # a valid pretrained_cfg argument takes priority over tag in model name
            pretrained_cfg = pretrained_tag

    if not is_model(model_name):
        raise RuntimeError("Unknown model (%s)" % model_name)

    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = create_fn(
            pretrained=pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            **kwargs,
        )

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, filter_keys=filter_keys, state_dict_map=state_dict_map)

    return model
