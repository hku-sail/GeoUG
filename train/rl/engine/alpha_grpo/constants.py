# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Prompt templates used across tasks."""

VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer.
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image.
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

reflection_and_regenerate_prompt_with_caption = """The user requires to generate image of `{}`. You tried to generate an image based on the user's request, but you failed to create the correct image.
Reflect on what went wrong and write down the editing instruction that will help you do better and refine the image to meet user's request based on your own reflection."""
