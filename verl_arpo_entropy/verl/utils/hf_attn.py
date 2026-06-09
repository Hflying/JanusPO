# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hugging Face ``attn_implementation`` selection when flash-attn is optional."""

import os


def get_hf_attn_implementation(default_without_flash: str = "sdpa") -> str:
    """Return a value valid for Transformers ``attn_implementation``.

    Uses ``flash_attention_2`` when the ``flash_attn`` package imports successfully;
    otherwise returns ``default_without_flash`` (PyTorch SDPA by default).

    Override anytime with env ``VERL_ATTN_IMPLEMENTATION`` (e.g. ``eager``, ``sdpa``,
    ``flash_attention_2``).
    """
    override = os.environ.get("VERL_ATTN_IMPLEMENTATION", "").strip()
    if override:
        return override
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return default_without_flash
