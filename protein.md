# AlphaProteo

首先感谢汪千桐对 AlphaFold 的分享.

由于作者在报告中已经提及,

> This report does not include machine learning methods due to biosecurity and commercial considerations.

> 出于生物安全和商业考虑, 本报告不包括机器学习方法.

我只能够根据少量信息以及实验方法, 数据, 和结论来进行推测, 给出一个简略的原理介绍. 这其中可能有很多不完备的地方.

## 概述

AlphaProteo 是 DeepMind 推出的一系列用于蛋白质设计的机器学习模型. 根据实验测试的结果, 相较于传统的蛋白质结合体设计, AlphaProteo 有这些特征

> 1. Highsuccess rate: stable, highly expressed, and specific binders can be obtained from screening tens of design candidates, alleviating the need for high-throughput methods.

> 2. High affinity: for every target tested except one, the best binders have sub-nanomolar or low-nanomolar binding affinity (KD), minimizing the labor needed for downstream affinity optimization.

> 3. General: binders are successfully obtained against a range of targets with diverse structural and biochemical properties, using a single design method without complex manual intervention.

RFdiffusion:

> 1. 高成功率: 通过筛选数十种候选设计, 可获得稳定、高表达和特异的结合体, 从而减少了对高通量方法的需求.

其他:

> 2. 高亲和力: 除一个靶点外, 对于每个测试靶点, 最佳结合体都具有亚纳摩尔或低纳摩尔的结合亲和力 (KD), 从而最大限度地减少了下游亲和力优化所需的人力.

> 3. 通用性: 使用单一设计方法, 无需复杂的人工干预, 即可成功获得针对一系列具有不同结构和生化特性的靶标的结合剂.

## 原理

原报告中真正讲述 AlphaProteo 的原理的似乎只有 Results 中的第一段

> AlphaProteo comprises two components (Figure 1A): a generative model trained on structure and sequence data from the Protein Data Bank (PDB) and a distillation set of AlphaFold predictions, as well as a filter which scores generated designs to predict whether they will succeed experimentally. To design binders, we input a structure of the "target" protein and optionally designate "hotspot" residues representing the target epitope; the generative model outputs a structure and sequence of a candidate binder for that target (Figure 1B). We generate a large number of design candidates and then filter them to a smaller set prior to experimental testing. The generative model compares favorably to the best existing method on in silico benchmarks (Figure S1, Section S2).

> AlphaProteo 包含两个组成部分 (图 1A): 一个生成模型, 基于蛋白质数据库 (PDB) 的结构和序列数据进行训练, 并结合 AlphaFold 的预测数据, 以及一个评分筛选器, 用于评估生成的设计是否具有实验成功的可能性. 为了设计结合体, 我们输入“目标”蛋白的结构, 并可选择指定代表目标抗原表位的“热点”残基; 生成模型输出该目标蛋白的候选结合体的结构和序列 (图 1B). 我们生成大量设计候选体, 然后将其筛选为一个较小的集合, 以便进行实验测试. 在计算机模拟测试中, 生成模型与现有最好的方法相比表现优异 (图 S1, 第 S2 节).

可见, AlphaProteo 由一个生成模型和一个筛选器组成.

### 生成模型

我们先介绍生成模型.

#### 设计和结构

生成模型可以大致分为三个部分, 其架构相当于一个条件生成网络.

> Schematic of target-structure-conditioned binder design as performed by the generative model.

> 生成模型进行的目标结构调整粘合剂设计示意图.

1. 生成网络: 通过机器学习模型生成新的蛋白质序列和结构, 确保这些设计符合特定的结合需求.

2. 目标条件约束: 模型根据目标蛋白结构中的热点区域生成候选结合体, 并将该结构作为生成的条件, 确保生成的结合体能够与目标位点形成有效结合.

除了基本的生成网络和条件约束, AlphaProteo 还有一个

3. 预测模块: 实时评估生成的候选体与目标蛋白的结合可能性.

下面介绍

#### 工作流程

1. 目标蛋白结构输入: 用户首先提供目标蛋白的三维结构数据, 这些结构数据可以来自实验测定 (如 X 射线晶体学和冷冻电镜) 或计算预测 (如 AlphaFold 模型预测). 生成模型将这些数据作为输入条件, 用于指导结合体生成.

2. 热点区域识别与设定: 结合目标蛋白的结构, 用户可以指定希望生成结合体的结合位点 (即热点区域). 这些热点区域是目标蛋白上具有功能性或特异性结合潜力的区域. 热点区域通常是表面残基或具有生物活性的位点, 例如抗体结合位点、酶的活性位点等. 生成模型会专注于与这些区域进行有效的相互作用.

3. 候选体的生成: 生成模型开始**从头**生成多个候选结合体序列和结构. 模型基于从蛋白质数据库 (PDB) 和 AlphaFold 预测数据中提取的特征, 结合目标蛋白的结构特点和热点区域设计出不同的结合体.

    1. 结构生成: 生成模型生成的蛋白质序列会自动预测其折叠后的三维结构. 这一步骤使用深度学习中的空间网络结构来实现, 确保生成的蛋白质具有稳定的三维折叠形式.

    2. 序列优化: 生成模型会对每个候选体的氨基酸序列进行优化, 以增强其与目标蛋白的结合亲和力. 模型会通过反复迭代和特征比对来细化这些生成的序列, 以确保候选结合体具有高效、稳定的结合能力.

##### 为什么从头设计?

> Computational design of binders de novo, without using a natural protein as a starting point, can target pre-specified epitopes and generate binders that are smaller, more thermostable, and easier to express than antibodies [10, 39, 6].

> 不使用天然蛋白质作为起点, 从头开始计算设计结合体, 可以针对预先指定的表位, 产生比抗体更小、更耐高温、更容易表达的结合体[10, 39, 6].

#### 训练数据和优化

1. 蛋白质数据库 (PDB) : 蛋白质数据库包含了大量通过实验手段测定的蛋白质三维结构, 生成模型可以从中提取不同蛋白质折叠方式、结合位点特性等多种结构特征.

2. AlphaFold 预测数据: 生成模型利用 AlphaFold 的预测数据补充 PDB 数据集, 尤其是对于那些尚未在实验中确定结构的蛋白质. AlphaFold 的高精度预测数据帮助模型适应更多未知结构的蛋白质, 从而提升了模型的广泛适用性.

3. 深度学习优化: 生成模型通过深度学习算法在上述数据集上进行训练, 提取不同蛋白质的结构和序列特征, 从而更精准地生成符合结合需求的蛋白质序列. 模型可能通过反复迭代和反馈优化来改进生成效果, 使生成的结合体结构更加符合目标蛋白的结合需求.

### 筛选器

现在简单介绍筛选器.

#### 核心功能

主要承担两个任务: 一是对生成模型生成的大量候选结合体进行评分, 二是根据评分结果筛选出高可能性成功的结合体, 从而减少实验测试的数量.

1. 结合体评分: AlphaProteo 的筛选模型会对生成模型生成的每个候选结合体进行评分. 评估生成的结合体是否符合与目标蛋白结合的生物物理和结构条件. 评分系统会结合目标蛋白的三维结构、结合体的表面特性以及结合位点的相似性, 为候选结合体分配一个分数.

2. 结合体的筛选与缩小范围: 生成模型通常会生成大量候选结构, 筛选模型则基于评分将候选体的数量大为缩减. 这些高分候选体被筛选后才进入实验验证. 通过筛选模型的过滤, AlphaProteo 能够从最初的大量候选体中筛选出可能性最高的部分, 显著减少后续的实验验证量.

#### 工作机制

1. 特征提取: 筛选模型会从每个候选体中提取结构和序列特征. 这些特征可能包括结合体与目标蛋白表面区域的适配度、潜在的结合力 (如氢键、疏水相互作用等), 以及蛋白质折叠的稳定性.

2. 评分计算: 根据提取的特征, 筛选模型使用评分函数对候选体进行打分. 评分函数由一系列加权特征组成, 这些特征会根据模型的训练数据进行学习和调整, 以确保对候选体的评分能够反映实际结合潜力.

3. 候选体筛选: 筛选模型对评分较高的候选体进行选择, 保留少数高分候选体用于实验验证. 这一过程显著减少了高通量筛选的需求, 使得 AlphaProteo 的实验流程更加高效.

## 实验

最后, 筛选模型输出的结合体候选结构在实验室中进行生物化学实验测试, 测量其结合强度 (如 KD 值) 和结合成功率. 由于其有一定难度, 这里就不单独介绍了, 有兴趣的同学可以去看原报告.

## 应用

下面将由剩下两位同学介绍 AlphaProteo 的实际应用.
