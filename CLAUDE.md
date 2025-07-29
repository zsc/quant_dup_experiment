（交流可以用英文，本文档中文，保留这句）

# 实验项目说明

## 项目目标
编写一份LLM 提速实验书markdown，每个大环节为一章
核心想法，是减少 <4bit weight 量化和 <16bit activation 量化时，LLM 的精度损失。
计划采取的主要补偿措施，是 duplicate some layers to compensate，即形成类似 layer k, layer k, layer k+1, layer k+1 这样的LLM'。当单层能装入计算设备内存（如 GPU 显存）时，第二次及以后的计算不需要再载入 weights，解决了带宽瓶颈问题。
主要实验对象是开源模型，或许包括 MoE。
组织为 index.md + chapter1.md + ...
