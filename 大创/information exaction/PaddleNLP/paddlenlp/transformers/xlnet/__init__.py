# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""
=========
模型简介

`XLNet: Generalized Autoregressive Pretraining for Language Understanding <https://arxiv.org/abs/1906.08237>`__
是一款无监督的自回归预训练语言模型。

有别于传统的单向自回归模型，XLNet通过最大化输入序列所有排列的期望来进行语言建模，这使得它可以同时关注到上下文的信息。
另外，XLNet在预训练阶段集成了 `Transformer-XL <https://arxiv.org/abs/1901.02860>`__ 模型，
Transformer-XL中的片段循环机制（Segment Recurrent Mechanism）和相对位置编码（Relative Positional Encoding）机制
能够支持XLNet接受更长的输入序列，这使得XLNet在长文本序列的语言任务上有着优秀的表现。

本项目是XLNet在 Paddle 2.0上的开源实现，由modeling和tokenizer两部分组成。
"""
