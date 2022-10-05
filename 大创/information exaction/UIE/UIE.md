# 论文简介
- [论文地址](https://arxiv.org/pdf/2203.12277.pdf)
- [github地址](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie)
- Unified Structure Generation for **Universal Information Extraction**（为通用信息抽取的统一结构生成）

## 研究背景
信息提取任务（Information Extraction,后文简称为IE）可以表述为文本到结构的问题，其中不同的 IE 任务对应不同的结构。
比如：
1. 实体
2. 关系
3. 时间关系
4. 情感分类
![](../../attachment/c5379a13a7f38333f732ea69da4e465.png)
而在信息抽取的多重任务的情况下，缺少一个统一的框架，这就导致了信息抽取的方法不能泛用，需要针对每一个任务独立开发一套独立的抽取方法。本文旨在通过单一框架统一建模不同 IE 任务的文本到结构转换，即不同的结构转换将在通用模型中共享相同的底层操作和不同的转换能力。形式上，给定特定的预定义模式 s 和文本 x，通用 IE 模型需要生成一个结构，该结构包含模式 s 指示的文本 x 中所需的结构信息
## 生成逻辑
为了解决上述的问题，UIE提出了如下的方法
1. structured extraction language (SEL) 
是一个通用的信息抽取语言，可以适用于所有的IE任务
	1.  SPOTNAME: 表示源文本中存在一个特定的信息片段，该类型的地点名称存在
	2. ASSONAME: 表示源文本中存在一个特定的信息片段，该片段与结构中其上层 Spotted 信息的 AssoName 关联
	3. INFOSPAN:表示与源文本中的特定定位或关联信息片段相对应的文本跨度。
	![|300](../../attachment/1663287567339.png)
> 上图为SEL的基本逻辑框架

![|300](../../attachment/1663287678845.png)
>  上图为分在在关系结构、事件结构、实体结构下的SEL语言结构
>  
1. structural schema instructor (SSI)
基于模式的提示机制，它控制 UIE 模型发现哪些、关联哪些以及为不同的提取设置生成结构
![](../../attachment/2a56828e88da13e8853e2ba95b3390d.png)
输入部分
$$
y=\operatorname{UIE}(s \oplus x)
$$
$$
\begin{aligned}
s \oplus x=& {\left[s_{1}, s_{2}, \ldots, s_{|s|}, x_{1}, x_{2}, \ldots, x_{|x|}\right] } \\
=& {[[\mathrm{spot}], \ldots[\mathrm{spot}] \ldots,} \\
& {[\operatorname{asso}], \ldots,[\operatorname{asso}] \ldots, } \\
& {\left.[\operatorname{text}], x_{1}, x_{2}, \ldots, x_{|x|}\right] }
\end{aligned}
$$
在上述的公式中，x表示的输入的文本，s表示SSI
例如，SSI“[spot] person [spot] company [asso] work for [text]”表示从句子中提取关系模式“the person works for the company”的记录。给定 SSI s，UIE 首先对文本 x 进行编码，然后使用编码器-解码器式架构在线性化 SEL 中生成目标记录 y

编码（encode）
$$
\mathbf{H}=\operatorname{Encoder}\left(s_{1}, \ldots, s_{|s|}, x_{1}, \ldots, x_{|x|}\right)
$$

解码（decode）
$$
y_{i}, \mathbf{h}_{i}^{d}=\operatorname{Decoder}\left(\left[\mathbf{H} ; \mathbf{h}_{1}^{d}, \ldots, \mathbf{h}_{i-1}^{d}\right]\right)
$$
## 模型的训练和生成过程

### 预训练
1. $\mathcal{D}_{\text {pair }}$ 
通过将 Wikidata 与英语 Wikipedia 对齐来收集大规模的并行文本结构对。 Dpair 用于**预训练 UIE 的文本到结构的转换能力**。
$$
\mathcal{D}_{\text {pair }}=\{(x, y)\}
$$
在上式中**x是UIE生成的结构**，**y是从 Wikidata等部分提取的结构**
$$
s_{+}=s_{\mathrm{s}+} \cup s_{\mathrm{a}+}
$$
在上式中$s_{\mathrm{s}+}$表示正确的spoting结果，$s_{\mathrm{a}+}$表示正确的assioantion结果
$$
s_{\text {meta }}=s_{+} \cup s_{\mathrm{s}_{-}} \cup s_{\mathrm{a}_{-}}
$$
除此之外,为了学习一般映射能力，我们还自动为每对**构建负模式**,例如，person and work for 是记录“((person: Steve (work for: Apple)))”中的正模式，我们可以选取vehicle和located in 为负模式来构建.
$$
\mathcal{L}_{\text {Pair }}=\sum_{(x, y) \in \mathcal{D}_{\text {pair }}}-\log p\left(y \mid x, s_{\text {meta }} ; \theta_{e}, \theta_{d}\right)
$$
上式为这个部分的损失函数，其中$\theta_{e}, \theta_{d}$为编码和解码的参数
2. $\mathcal{D}_{\text {record }}$ 
$$
\mathcal{L}_{\text {Record }}=\sum_{y \in \mathcal{D}_{\text {record }}}-\log p\left(y_{i} \mid y_{<i} ; \theta_{d}\right)
$$
3. $\mathcal{D}_{\text {text}}$ 
$$
\mathcal{L}_{\mathrm{Text}}=\sum_{x \in \mathcal{D}_{\text {text }}}-\log p\left(x^{\prime \prime} \mid x^{\prime} ; \theta_{e}, \theta_{d}\right)
$$
为了缓解语义的灾难性的遗忘问题


### 生成和微调
如果为了适应某一个任务
$$\mathcal{L}_{\mathrm{FT}}=\sum_{(s, x, y) \in \mathcal{D}_{\text {Task }}}-\log p\left(y \mid x, s ; \theta_{e}, \theta_{d}\right)$$
拒绝机制（Rejection  Mechanism）
![](../../attachment/a73eed758456880de7847c6a23bc43a.png)

- 为了减少自适应模型中的暴露偏差（exposure bias）

## 实验结果
![](../../attachment/16edb707c4a6e98c8a475e4287683c6.png)
> 上图上为在四个知识抽取任务13和数据集上的结果

![](../../attachment/e0dd202f82b2cc501252d8e152ce553.png)
> 上图为在低资源情况下的实验结果


![](../../attachment/96f63084a04d28a10aa6ad593cefc0c.png)
> 测试预训练的结果
![](../../attachment/267599477658c2428ae961d46e7f7d5.png)
>测试RM的效果









$$\mathcal{L}_{\mathrm{FT}}=\sum_{(s, x, y) \in \mathcal{D}_{\text {Task }}}-\log p\left(y \mid x, s ; \theta_{e}, \theta_{d}\right)$$
拒绝机制（Rejection  Mechanism）
![](../../attachment/3b16abca843d8068f74ad9e421d1005.png)
为了减少在自适应模型中的暴露偏差（exposure bias）


