（交流可以用英文，本文档中文，保留这句）

# 实验项目说明

## 项目目标
编写一份LLM 提速实验书markdown，每个大环节为一章
核心想法，是减少 <4bit weight 量化和 <16bit activation 量化时，LLM 的精度损失。
目标是实现 2-bit weight 量化和 8-bit 或 16-bit activation 量化，在 M1 Pro 上运行 32B 模型时保持极小的 PPL 损失。
计划采取的主要补偿措施，是 duplicate some layers to compensate，即形成类似 layer k, layer k, layer k+1, layer k+1 这样的LLM'。当单层能装入计算设备内存（如 GPU 显存）时，第二次及以后的计算不需要再载入 weights，解决了带宽瓶颈问题。
核心动机是提高计算访存比。基础形式是将层复制一次形成两个相邻的共享权重层。更高级的形式可能为这两层使用不同的 scaling factors（以及其他量化超参数）以减少量化误差。
主要实验对象是开源模型：Qwen 系列（密集模型）和 DeepSeek（MoE 模型）。
组织为 index.md + chapter1.md + ...

## 技术约束与设计原则

### 量化参数调整
- 共享权重的层可以有不同的 scaling factor 和 zero point（可学习或固定）
- 可调整的参数包括：activation zero point & scaling factor, group size，以及其他量化超参数

### 集成方法约束
- 已知 <2-bit 量化损失严重，考虑集成方法帮助恢复能力
- 约束1：集成的多个模型必须共享权重
- 约束2：最小化训练工作量（如需要 QAT，做最少的训练）

### MoE 特殊处理
- 不关注推理时的负载均衡
- 重点关注低比特量化下受损严重的重要 experts
- Router/gate 计算成本低，不进行量化

### 评估指标
- 质量指标：主要使用 PPL
- 性能指标：记录延迟和带宽（预期略有下降，因为用计算换取权重加载带宽）

### 实现框架
- GPTQ/AWQ 作为可替换的插件
- 本实验书不包含具体代码实现
