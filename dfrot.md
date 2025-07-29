# Document 2: DFRot: Achieving Outlier-Free and Massive Activation-Free for Rotated LLMs with Refined Rotation

**arXiv:2412.00648v4 [cs.LG] 15 Jul 2025**
*Published as a conference paper at COLM 2025*

## DFRot: Achieving Outlier-Free and Massive Activation-Free for Rotated LLMs with Refined Rotation

**Jingyang Xiang**
New York University
xiangxiangjingyang@gmail.com

**Sai Qian Zhang**
New York University
sai.zhang@nyu.edu

### Abstract

Rotating the activation and weight matrices to reduce the influence of outliers in large language models (LLMs) has recently attracted significant attention, particularly in the context of model quantization. Prior studies have shown that in low-precision quantization scenarios, such as 4-bit weights and 4-bit activations (W4A4), randomized Hadamard transforms can achieve significantly higher accuracy than randomized orthogonal transforms. Notably, the reason behind this phenomenon remains unknown. In this paper, we find that these transformations show substantial improvement in eliminating outliers for common tokens and achieve similar quantization error. The primary reason for the accuracy difference lies in the fact that randomized Hadamard transforms can slightly reduce the quantization error for tokens with massive activations while randomized orthogonal transforms increase the quantization error. Due to the extreme rarity of these tokens and their critical impact on model accuracy, we consider this a long-tail optimization problem, and therefore construct a simple yet effective method: a weighted loss function. Additionally, we propose an optimization strategy for the rotation matrix that involves alternating optimization of quantization parameters while employing orthogonal Procrustes transforms to refine the rotation matrix. This makes the distribution of the rotated activation values more conducive to quantization, especially for tokens with massive activations. Our method enhances the Rotated LLMs by achieving dual free, *Outlier-Free* and *Massive Activation-Free*, dubbed as DFRot. Extensive experiments demonstrate the effectiveness and efficiency of DFRot. By tuning the rotation matrix using just a single sample, DFRot achieves a perplexity improvement of 0.98 and 0.95 on W4A4KV4 and W4A4KV16, respectively, for LLaMA3-70B, a model known for its quantization challenges. Code is available at https://github.com/JingyangXiang/DFRot.

### 1 Introduction

Large Language Models (LLMs) have shown exceptional abilities across numerous domains. Cutting-edge open-source models like LLaMA (Touvron et al., 2023) and Mistral (Jiang et al., 2023), along with proprietary LLMs such as GPT (Achiam et al., 2023) and Gemini (Team et al., 2023), are now being applied in a wide range of applications, including natural language understanding (Zellers et al., 2019; Hendrycks et al., 2020), machine translation (Zhang et al., 2023), content generation (Mo et al., 2024), recommendation systems (Wu et al., 2023; Wang et al., 2024; 2025) and agent (Li et al., 2025).

However, the remarkable success of LLMs is largely reliant on significant computational resources. LLMs often consist of billions of parameters, making them not only resource-intensive to train but also challenging to deploy on devices with limited computational capacity, such as mobile phones and edge devices. Additionally, the high memory and processing demands not only drive up hardware costs but also significantly increase energy consumption, leading to serious deployment concerns. To address these challenges, researchers and engineers are actively exploring various model compression techniques (Frantar et al., 2022; Xiao et al., 2023; Lin et al., 2024a; Yao et al., 2022; Frantar & Alistarh, 2023; Ashkboos et al., 2024a; Wei et al., 2024; Zhao et al., 2025). These techniques aim to reduce the size of LLMs while maintaining their performance as effectively as possible, achieving a balance between efficiency and accuracy.

Unfortunately, the presence of outliers in the activations (Dettmers et al., 2022; Zeng et al., 2022) often leads to a significant reduction in model accuracy when PTQ is applied directly. To address this problem, earlier approaches have either scaled weights and activations (Xiao et al., 2023; Wei et al., 2023; Shao et al., 2023), shifting the quantization challenges from activations to weights, or employed mixed-precision techniques to isolate outliers (Dettmers et al., 2022), thereby minimizing the LLM's quantization error.

Recent research (Ashkboos et al., 2024b) has demonstrated that rotating activations in LLMs can effectively eliminate most outliers while preserving computational invariance, ensuring that the LLM's output remains identical to its original results. Moreover, the rotation matrices can be merged into the weights, imposing no additional burden on network inference. This innovative computational invariance (Ashkboos et al., 2024a) has garnered significant attention from researchers.

Although rotation is widely recognized as an important method for the quantization of LLMs, there remain many unresolved issues. For example, as shown in Table 1, when activations are reduced to 4-bit, the reasons why randomized Hadamard transforms (RH) often achieve significant improvement compared to randomized orthogonal transforms (RO) (Ashkboos et al., 2024b; Liu et al., 2024) have not yet been fully understood. However, while directly training rotation matrices can yield good results (Liu et al., 2024), the training process will cause substantial computational resources and adds complexity to the quantization process.

In this paper, we first investigate the underlying reasons why RH outperforms RO. We find that for ordinary tokens consisting primarily of outliers (Achiam et al., 2023), both RO and RH transformations can equally reduce quantization error when applied to these tokens. As shown in Figure 3, in terms of quantization error, there is no substantial difference between the two transformations. In contrast, for special tokens with massive activations (Sun et al., 2024), using RO on these activations surprisingly leads to an increase in quantization error. Our experiments show that this inability to efficiently manage massive activations greatly restricts the accuracy of quantized LLMs. On the other hand, while RH performs better than RO, it only manages to maintain or slightly reduce the quantization error for these large activations. This observation indicates that both transformation methods struggle to effectively manage massive activations in LLM quantization.

Building on these insights, we propose a novel optimization method to enhance the performance of quantized LLMs, achieving both *Outlier-Free* and *Massive Activation-Free*, e.g. dual free (DFRot). By treating scarce tokens with massive activations as long-tail distributed data, we develop a simple yet effective weighted loss function. Additionally, we introduce an alternating optimization approach to refine the rotation matrices and quantization parameters, further minimizing quantization error. Extensive experiments demonstrate the effectiveness of our proposed method. Specifically, by tuning the rotation matrix with just a single sample, DFRot achieves a PPL improvement of 0.95 and 0.98 on W4A4KV4 and W4A4KV16 for LLaMA3-70B with WikiText-2, a model recognized for its quantization challenges (Huang et al., 2024).

### 2 Related Work

#### 2.1 Eliminating outliers via Scale Invariance

The initial idea behind suppressing outliers through scale invariance stems from the observation that weights are easier to quantize than activations, and outliers in activations often appear in a few fixed channels Dettmers et al., 2022. Based on this, SmoothQuant (Xiao et al., 2023) first proposes that we can offline migrate the quantization difficulty from activations to weights via scale invariance. SmoothQuant enables an INT8 quantization of both weights and activations for all the matrix multiplications in LLMs. Furthermore, Outlier Suppression+ (Wei et al., 2023) proposes a fast and stable scheme to effectively calculate scaling values, achieving a better balance in quantization burden. To reduce manual design and further enhance quantization performance in extremely low-bit quantization, Omni-Quant (Shao et al., 2023) introduces Learnable Weight Clipping and Learnable Equivalent Transformation, efficiently optimizing the quantization process for both weight-only and weight-activation quantization. In the clipping W4A8 quantization, QQQ (Zhang et al., 2024) proposes to dynamically handle outliers through adaptive smoothing. QServe (Lin et al., 2024b) proposes SmoothAttention to effectively mitigate the accuracy degradation caused by 4-bit KV quantization. Both QQQ and QServe have effectively enhanced the performance of LLMs in W4A8 quantization.

#### 2.2 Eliminating outliers via Rotational Invariance

Although scale invariance can reduce outliers and improve quantization performance, it merely transfers the outliers from activations to weights and has not eliminated them fundamentally. When the magnitude of the outliers is large, scaling struggles to achieve an effective balance between weights and activations. Recently, researchers have found that applying rotation matrices to networks can effectively reduce outliers without increasing the complexity of LLMs. QuIP Chee et al. (2024) is the first to suggest that quantization can benefit from the incoherence between weight and Hessian matrices. It employed randomized orthogonal matrices generated by Kronecker product to enhance their incoherence. QuIP# (Tseng et al., 2024) replaces the randomized orthogonal matrices with randomized Hadamard matrices, which are faster and possess better theoretical properties. QuaRot (Ashkboos et al., 2024b) is the first work to apply rotational invariance (Ashkboos et al., 2024a) for model quantization. QuaRot finds that randomized Hadamard transformations yield better results compared to randomized orthogonal transformations. SpinQuant (Liu et al., 2024) and OSTQuant (Hu et al., 2025) further extends the rotation matrices to a trainable space and applied Cayley optimization (Li et al., 2020) to refine them, achieving significant improvements across diverse datasets.

### 3 Rotational Invariance, Quantization and Massive Activation

#### 3.1 Rotational Invariance

First, we briefly introduce rotational invariance in LLMs, using the structure of LLaMA as an example. We assume that the α in the RMSNorm has been fused into the follow linear layers' weights, including Wq, Wk, Wv, Wup and Wgate and RMSNorm applies to each row of the activations X as X_i,: ← X_i,: / ||X_i,:||. If R₁ is an rotation matrix, we have the commutation property RMSNorm(XR₁) = RMSNorm (X) R₁ (Ashkboos et al., 2024a). This property implies that multiplying the input of RMSNorm by R₁ is equivalent to multiplying the RMSNorm output by R₁.

---

**Figure 1: An illustration of rotational invariance in the LLaMA architecture. The rotation matrix R₁ can be integrated into the residual connection, ensuring the network retains rotational invariance. The rotation inner the block can further reducing outliers within block. Both of them make LLM fewer outliers and be easier to quantize. The rotation matrix R₁, R⁻¹₁, R₂, R³ and R⁴ can be integrated into weights. R₃ and R₄ need to compute online.**
---

As shown in Figure 1, to remove outliers in the input activations, a rotation matrix R₁ is applied to the embedding layer W_embedding, resulting in a new input activation X₁R₁. According to the above, we can know once we transform Wq, Wk, Wv, and Wo in the Multi-Head Attention (MHA) to Rᵀ₁Wq, Rᵀ₁Wk, Rᵀ₁Wv and WₒR₁, the hidden feature within the MHA will remain unchanged, and the original output feature Y₁ will become Y₁R₁. The following Feed-Forward Network's input X₂ from the residual connection will be modified to (X₁ + Y₁)R₁ = X₂R₁. If we further transform W_up, W_gate and W_down to Rᵀ₁W_up, Rᵀ₁W_gate and W_downR₁, the hidden feature within the FFN will also remain unchanged, and the output feature X₃ will be modified to (X₂ + Y₂)R₁ = X₃R₁. Based on mathematical induction, we can get that XₙR₁ + YₙR₁ = (Xₙ + Yₙ)R₁ = Xₙ₊₁R₁ for the n-th module. To this end, by transforming W_lm_head into Rᵀ₁W_lm_head, the network output will remain unchanged.

There is also rotational invariance within the block. For MHA, we can insert head-wise rotation matrices R₂ and R⁻¹₂ for Wᵥ and Wₒ and R₃ for Query and Key after RoPE. For FFN, we can insert R₄ and R⁻¹₄ between Swish and W_down. These approaches can further eliminate outliers and reduce quantization error while keeping the block output unchanged. In this paper, we only discuss R₁. For R₂, R₃, and R₄, we follow the QuaRot (Ashkboos et al., 2024b) settings and use Hadamard matrices.

#### 3.2 Why the Randomized Hadamard is better than Randomized Orthogonal?

Based on the computational invariance described in Section 3.1, it is evident that the choice of rotation matrices is critical for ensuring the accuracy performance of the quantized model. Therefore, a natural question arises: What type of rotation matrix offers the most advantageous properties?

---
**Table 1: WikiText-2 perplexity (↓) results for RO and RH for LLaMA models. The 4-4-4, 4-4-16, 4-8-16 represent W4A4KV4, W4A4KV16, W4A8KV16 respectively. We show the failed GPTQ using NaN and the perplexity results>100 by Inf. QuaRot.FP16() denotes retaining tokens with massive activations as FP16.**

| Method | LLaMA-7B 4-4-4 | LLaMA-7B 4-4-16 | LLaMA-7B 4-8-16 | LLaMA2-7B 4-4-4 | LLaMA2-7B 4-4-16 | LLaMA2-7B 4-8-16 | LLaMA2-13B 4-4-4 | LLaMA2-13B 4-4-16 | LLaMA2-13B 4-8-16 | LLaMA3-8B 4-4-4 | LLaMA3-8B 4-4-16 | LLaMA3-8B 4-8-16 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPTQ | NaN | NaN | NaN | NaN | NaN | NaN | Inf | Inf | 6.01 | Inf | Inf | 7.29 |
| (RO) QuaRot | 6.68 | 6.62 | 5.80 | 7.96 | 7.71 | 5.61 | 6.00 | 5.92 | 4.99 | 10.54 | 10.15 | 6.52 |
| (RO) QuaRot.FP16() | 6.30 | 6.27 | - | 6.17 | 6.10 | - | 5.38 | 5.34 | - | 7.83 | 7.68 | - |
| (RH) QuaRot | 6.37 | 6.33 | 5.81 | 6.27 | 6.20 | 5.61 | 5.51 | 5.46 | 5.01 | 8.20 | 8.02 | 6.52 |
| (RH) QuaRot.FP16() | 6.30 | 6.28 | - | 6.17 | 6.10 | - | 5.40 | 5.37 | - | 7.82 | 7.67 | - |
---

We begin by focusing on RO and RH, as both QuaRot (Ashkboos et al., 2024b) and Spin-Quant (Liu et al., 2024) have demonstrate that RH delivers substantial improvements over RO in LLMs. We conducted experiments by applying RO and RH to the LLaMA models respectively, followed by weight quantization using GPTQ under various quantization settings. The results are shown in Table 1. Benefiting from the outlier elimination through rotational invariance, we find that for dynamical token-wise 8-bit activation quantization, both RO and RH lead to significant performance improvements compared to standard quantization. Additionally, no substantial performance difference is observed between the two transformations. However, under 4-bit dynamical token-wise activation quantization, RH significantly outperforms RO.

To investigate the performance differences between RH and RO under 4-bit activation setting, we plot the corresponding quantization error after applying 4-bit quantization to the multiple tokens. We also display the quantization error for the baseline setting where quantization is applied without rotating the activation to better understand the impact of using the rotation matrix. As shown in Figure 2, compared to the no rotation (NR), both RO and RH effectively reduce the quantization error for most tokens across different models. While RH slightly lowers the quantization error, the difference between the two methods is minimal for the majority of tokens. This leads to the question: **What explains the significant difference in PPL during quantization when their quantization errors are so similar?**

To answer this question, we turn our attention to massive activation (Sun et al., 2024), a rare but significant feature in LLMs. As shown in Figure 2, the red points represent quantization error for the tokens with massive activation. While most tokens show large quantization errors under NR, these special tokens display significantly smaller errors, which can be observed from Figure 3. It is normal since each token has a fixed L2 norm after RMSNorm processing, as shown in Figure 4(a), tokens with massive activation naturally exhibit smaller quantization errors when quantized to 4-bit. Figure 4 presents the quantization result for the token with massive activation after applying NR, RO, and RH. Surprisingly, the rotation operations do not significantly reduce quantization errors for these tokens. In fact, compared to NR, RO greatly increases their quantization error, while RH only marginally reduces it. This leads us to question whether tokens with massive activation are the primary cause of the significant accuracy discrepancies between RH and RO.

To investigate this further, we build upon QuaRot by retaining tokens with massive activations in FP16 format for both RO and RH, while applying 4-bit quantization to the remaining input tokens, denoted as (RO) QuaRot.FP16() and (RH) QuaRot.FP16(). As shown in Table 1, for all LLaMA models, the performance gap between RH QuaRot and RO QuaRot is totally disappeared. It is so surprising that by simply retaining these extremely few tokens (often less than one-thousandth) as FP16, we can completely eliminate the performance difference between RO and RH. Therefore, we can make the following conclusion:

> **Why the Randomized Hadamard is better than Randomized Orthogonal?**
>
> RH ≈ RO + Tokens with Massive Activations: RH is better than RO because it performs more effectively when reducing the quantization error for tokens with massive activations in 4-bit activation quantization.

#### 3.3 Optimization Objectives and Calibration Data Selection

As mentioned above, although retaining tokens with massive activations as high-precision floating-point numbers can significantly enhance model accuracy, this approach is akin to a token-level version of LLM.int8(). It still requires fine-grained mixed-precision computations during the process, which will introduce additional system level optimization. Therefore, in this paper, we focus on W4A4 quantization to maintain simplicity and efficiency in the computation process. We consider a loss function of the following form:

L(R₁, gₓ) = Eₓ [||xR₁ - Q(xR₁, gₓ) ||²], (1)

where x ∈ ℝ¹ˣC is the token vector from a calibration dataset X^cal ∈ ℝ^LˣC. C is the hidden size and L is the number of tokens. R₁ ∈ ℝ^CˣC satisfies R₁Rᵀ₁ = I, gₓ is the quantization parameters and Q(x, gₓ) ∈ ℝ¹ˣC is the quantization of the x. The expectation Eₓ[·] is taken over the token distribution. For the ease of analysis, we use the mean squared error ||·||².

Meanwhile, we introduce our data selection principle. We denote the calibration dataset as X, the tokens with massive activations as Xᵐ, and the remaining tokens as X \ Xᵐ:

L(R₁, gₓ) = Eₓ∈X^cal\Xᵐ [||xR₁ - Q(xR₁, gₓ) ||²] + γ²Eₓ∈Xᵐ [||xR₁ - Q(xR₁, gₓ)||²]. (2)

During calibration, we apply a weighted loss to prioritize the quantization error on tokens with massive activations, with γ representing the weight.

The motivation behind this principle stems from the observations in Table 1. Since Xᵐ is the key factor contributing to the performance gap between RO and RH, simply optimizing R₁ via Eq. 1 fails to specifically target Xᵐ. On the other hand, compared to the NR in Table 1, RO also significantly improves performance, indicating that reducing the outliers on X^cal \ Xᵐ can enhance quantization performance, optimizing only for Xᵐ has the risk of increasing the quantization error for X^cal \ Xᵐ, ultimately degrading the model's performance. Hence, it is crucial to optimize both Xᵐ and X^cal \ Xᵐ. Naturally, we can regard this a long-tail optimization problem, where Xᵐ represents the long-tail but important data. Using a weighted approach to optimize the quantization loss is a simple yet highly effective method. Ablation studies in Section 4.2 further demonstrate the advantages of this strategy.

#### 3.4 Solution Methods

Optimizing R₁ is a challenging task. Since R₁ influences every MHA and FFN in the network, adjusting the activation distribution in one layer impacts the quantization results across all layers. This makes it difficult to optimize layer by layer or block by block (Shao et al., 2023; Wei et al., 2023). A straightforward approach is to use training methods for quantization-aware fine-tuning of the rotation matrix across the entire network (Liu et al., 2024). Although it does not require retaining the gradients of the weights or the corresponding optimizer states, it still demands substantial computational resources during the quantization process.

In this paper, we focus on improving the effectiveness of rotation matrices in mitigating outliers and massive activation. Intuitively, we hypothesize that a rotation matrix that minimizes quantization error will lead to better performance. Drawing inspiration from Simsiam (Chen & He, 2021), we propose to regard quantization representation Q(xR₁, g) as cluster centroids ηₓ. In the context, optimizing R₁ and g is equivalent to optimizing R₁ and ηₓ, which can be viewed as an implementation of an Expectation-Maximization (EM)-like algorithm, as shown in the following equation:

min_{R₁,ηₓ} L(R₁,ηₓ) = Eₓ∈X^cal\Xᵐ [||xR₁ - ηₓ||²] + γ²Eₓ∈X^cal [||xR₁ - ηₓ||²]
= Eₓ∈X̃^cal [||x̃R₁ - ηₓ||²], (3)

where ηₓ = Q(xR₁,g) and X̃^cal = {x|x ∈ X^cal \ Xᵐ} ∪ {γx|x ∈ Xᵐ}. This formulation is analogous to k-means clustering (Macqueen, 1967), and R₁ and ηₓ act like the kernel function and cluster centroids, respectively. Similar to k-means clustering, the problem described in Eq 3 can be approached using an alternating algorithm, where one set of variables is fixed while solving for the other. Formally, we can alternate between solving these two subproblems:

ηₓᵗ ← arg min_{ηₓ} L(R₁ᵗ⁻¹, ηₓ); R₁ᵗ ← arg min_{R₁} L(R₁, ηₓᵗ) (4)

where t represents the iteration index of the alternating rounds, and ηₓᵗ and R₁ᵗ denote the values of ηₓ and R₁ at round t.

**Solving for the cluster centroids ηₓ.** The set of quantization parameters gₓ further contains the quantization scale sₓ and zero point zₓ. In this paper, we adopt dynamic asymmetric per-token quantization for activations. Therefore, we can independently determine the optimal quantization scheme for solving sₓ and zₓ for each xR₁:

ηₓ = Q_g(xR₁, sₓ, zₓ) = clamp([xR₁/sₓ] + zₓ, 0, 2ᴺ - 1),
where sₓ = (max(xR₁) - min(xR₁)) / (2ᴺ - 1), zₓ = -[min(xR₁)/sₓ] (5)

where [·] indicates round operation, N is the bitwidth.

**Solving for R₁.** The right side of Eq 4 is well-known as Procrustes problem (Mulaik, 2009), which involves finding the optimal rotation matrix R₁ that best aligns two sets of points, minimizing the Frobenius norm of their difference. The solution to this problem can be obtained through Singular Value Decomposition (SVD). Specifically, given input matrices X and its quantized version Q(X, gₓ), the optimal R₁ can be found:

R₁ = UVᵀ, where U, Σ, Vᵀ = SVD(X̃ᵀQ(X̃, gₓ)). (6)

where we treat the quantization parameters gʻ as a constant.

**One-step optimization.** To find an improved rotation matrix R₁ and quantization parameters gₓ, we perform the iterative process shown in Eq 4. Specifically, a calibration set X^cal is randomly sampled from X, the iterative process can be specified as:

sₓᵗ, zₓᵗ ← arg min_{sₓ,zₓ} Σ_{x∈X̃^cal} [||xR₁ᵗ⁻¹ - Q_{sₓ,zₓ}(xR₁ᵗ⁻¹)||²], ηₓᵗ ← Q_{sₓᵗ,zₓᵗ}(xR₁ᵗ⁻¹), (7)

then the resulting quantization parameters will be used to produce the rotation matrix:

R₁ᵗ ← arg min_{R₁} Σ_{x∈X̃^cal} [||xR₁ - ηₓᵗ⁻¹||²] (8)

The detailed algorithm is provided in Algorithm 1.

### 4 Experiments

**Experiment settings.** We implemented DFRot based on QuaRot. In this paper, to simplify the problem, we apply dynamic asymmetric per-token quantization for activation values. The KV-cache is quantized using asymmetric quantization with a group size of 128. GPTQ (Frantar et al., 2022) are used for weight with per-channel symmetric quantization, where a linear search for the clipping ratio is applied to minimize squared error. We use a sample with sequence length of 2048 from WikiText-2 (Merity et al., 2016) training set to generate calibration dataset X^cal, initialize the rotation matrix R₁ with RH, and optimize it for 100 iterations. After obtaining the optimized rotation matrix R₁, we apply it to the corresponding model and achieve rotational invariance. We use 128 samples each with a sequence length of 2048, as the calibration dataset for GPTQ quantization.

#### 4.1 Main results

**Language Generation Task.** We evaluate DFRot on a language generation task and compare it with SmoothQuant (Xiao et al., 2023), GPTQ (Frantar et al., 2022), OmniQuant (Shao et al., 2023), AWQ (Lin et al., 2024a), SpinQuant (Liu et al., 2024) and OSTQuant (Hu et al., 2025). Table 2 shows the perplexity of LLaMA models. As shown, compared to QuaRot, DFRot achieves improvements in most cases. For example, DFRot achieves the most significant improvement on the LLaMA3-8B model with W4A4KV4 and W4A4KV16, outperforming QuaRot by 0.25 and 0.21, respectively. It is worth noting that DFRot has achieved near 1.00 PPL improvement on LLaMA3-70B, a model known for its challenging quantization performance, even surpassing SpinQuant, which finetunes R₁ on wikitext through quantization-aware-training.

Similar to QuaRot, DFRot does not require any retraining process and only needs a sample to optimize the rotation matrix. On a single NVIDIA A100 80G GPU, it only takes an extra 8 minutes for LLaMA-7B & LLaMA2-7B & LLaMA3-8B and 20 minutes for LLaMA2-13B, resulting in minimal overhead. Even for the 70B models, the additional time is less than 90 minutes, which is also acceptable. It demonstrates that DFRot has wide applicability and can serve as a cost-effective post-training method to enhance the quantization performance of rotated LLMs. Although DFRot does not achieve the best performance compared to the state-of-the-art methods, like OSTQuant, we believe DFRot also help community to understand the fundamental performance gap between RO and RH.

**Zero-Shot Tasks.** We also evaluate DFRot on the following nine important zero-shot tasks: BoolQ (Clark et al., 2019), PIQA (Bisk et al., 2020), WinoGrande (Sakaguchi et al., 2021), OpenBookQA (Mihaylov et al., 2018), SIQA (Sap et al., 2019), HellaSwag (Zellers et al., 2019), Arc (Easy and Challenge) (Clark et al., 2018) and LAMBADA (Radford et al., 2019). We use lm_eval==0.4.5 (Gao et al., 2024) or our experiments. Table 2 shows the average score of DFRot on the above tasks. As can be seen, DFRot consistently achieves improvements compared to QuaRot across all tasks. For example, DFRot achieves a 1.56% accuracy improvement compared to QuaRot on the LLaMA3-8B model with W4A4KV4 quantization settings.

#### 4.2 Ablation studies 

**Choice of γ.** To further understand the effect of hyperparameters in DFRot, we conducted an ablation study on Wikitext-2 PPL to investigate the impact of different γ settings for W4A4KV16. As seen in Figure 5, when γ ranges between 50 and 200, DFRot achieves significant improvements across various LLaMA models using RH. Notably, on the LLaMA3-8B model, which is known for its quantization performance sensitiveness to massive activations from Table 1, we observed a PPL improvement of over 0.2 in Figure 5(d). If we set γ = 1 and treat Xᵐ and Xᶜᵃˡ \ Xᵐ equally to minimize their quantization errors, it may reduce the quantization loss of Xᶜᵃˡ \ Xᵐ but increase the quantization loss of Xᵐ, ultimately resulting in a performance decline on the LLaMA2-13B. Conversely, if we set γ → ∞ and only optimize the quantization error for Xᵐ, it will increase the quantization error of Xᶜᵃˡ \ Xᵐ, resulting in an accuracy drop across the LLaMA-7B, LLaMA2-7B, LLaMA2-13B and LLaMA3-8B models.

**Initialize with Randomized Orthogonal.** We conducted an ablation to study the effectiveness of DFRot when R₁ initialized with RO. We keep the same experimental settings as in the study with RH and optimize the rotation matrix with different γ values. As shown in Figure 6, our method achieves considerable improvements in RO scenarios compared to using RH for initialization. Meanwhile, it is more effective for LLM whose quantization performance is more sensitive to the massive activations, such as LLaMA3-8B and LLaMA3-70B. However, due to the exceptional performance of RH, initialization and optimization using RH always yield superior final results compared to those obtained with RO. Therefore, we recommend using RH for initialization in practice to achieve better performance.

**Figure 5: Comparison of WikiText-2 perplexity results under different γ for W4A4KV16. R₁ is initialized with RH.**
*(图表内容为曲线图，此处省略)*

**Figure 6: Comparison of WikiText-2 perplexity results under different γ for W4A4KV16. R₁ is initialized with RO.**
*(图表内容为曲线图，此处省略)*

**Table 3: Comparison of WikiText-2 perplexity results under different calibration samples for W4A4KV16.**

| Model | Sample1<br>(64×2048) | Sample2<br>(64×2048) | Sample3<br>(64×2048) | Sample4<br>(64×2048) | Sample5<br>(64×2048) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LLaMA3-8B | 7.78 | 7.76 | 7.79 | 7.74 | 7.76 |
| LLaMA2-7B | 6.13 | 6.12 | 6.15 | 6.11 | 6.14 |
| **Model** | **Sample1**<br>**(48x2048)** | **Sample1**<br>**(32×2048)** | **Sample1**<br>**(24×2048)** | **Sample1**<br>**(16×2048)** | **Sample1**<br>**(8×2048)** |
| LLaMA3-8B | 7.78 | 7.78 | 7.77 | 7.80 | 7.86 |
| LLaMA2-7B | 6.14 | 6.15 | 6.15 | 6.12 | 6.20 |

#### 4.3 Analysis of Calibration Set Sensitivity

We performed ablation studies W4A4KV16 on LLaMA3-8B and LLaMA2-7B along two dimensions: the choice of calibration samples and the num of calibration tokens and evaluate results on WikiText. Samples are all sampled from WikiText-2 train. In selecting the number of tokens, we utilize the inputs to the first N transformer blocks as the calibration data source and demonstrate results in Table 3. For example, when 48×2048 tokens are selected, the inputs to transformer blocks 0 to 23 are used for calibration. Our results indicate that, for tuning the rotation matrices in LLaMA3-8B and LLaMA2-7B, using 16×2048 tokens is often sufficient. We believe that these reasons may all be the causes for the optimization of DFRot being relatively insensitive to the number of tokens in the calibration dataset:

1.  These special tokens will appear in relatively shallow network layers (Sun et al., 2024), therefore, a small number of layers are also sufficient to capture these tokens.
2.  For a model, tokens with massive activations in LLMs tend to exhibit only a few similar data distributions because these tokens are often produced by out_proj or down_proj layers with large weights (Yu et al., 2024).
3.  GPTQ will use 128 samples with a length of 2048 to calibrate the weights, which reduces the impact of the sample size during rotation matrix calibration.

#### 4.4 Results on MMLU

**Table 4: Comparison of MMLU results under different methods.**

| W-A-KV | Methods | LLaMA2-7B | LLaMA3-8B | QWen2-7B | Mistral-7B-v0.3 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 16-16-16 | FP | 41.85 | 62.23 | 69.47 | 59.11 |
| 4-4-16 | QuaRot | 34.83 | 51.43 | 62.67 | 52.82 |
| 4-4-16 | DFRot | 35.54 | 51.68 | 63.40 | 53.38 |

We compare DFRot with QuaRot with W4A4KV16 quantization configuration with different models. As seen in Table 4, even though rotation matrix R₁ is refined with WikiText-2 dataset, DFRot also outperforms QuaRot in all models. It indicates that DFRot, which refines R₁ by optimized long tailed quantization error, can be seen as a general method. It is also worth noting that even though DFRot achieves slight improvement with WikiText2 for LLaMA2-7B, it achieves 0.71% improvement with MMLU, which is significant. On the contrary, for the LLaMA3-8B, while DFRot achieves significant improvement with WikiText2, it only achieves 0.25% improvement with MMLU, which is slight. To sum up, we can know that the PPL with WikiText2 can not been seen as a good indicator of the model downstream performance. In the future, we will study how to design more robust quantization algorithms for downstream tasks to further enhance the capabilities of quantized models in downstream tasks.

### 5 Conclusion

Eliminating outliers in LLMs through rotational invariance can significantly improve model quantization accuracy. In this paper, we find that in the context of 4-bit activation quantization, the fundamental reason for the effectiveness difference between RO and RH is their performance on tokens with massive activations. Specifically, randomized Hadamard transformations perform better on these tokens than random Orthogonal transformation. Based on the observation that tokens with massive activations are rare and important in LLMs, we treat the problem as a long-tail optimization and construct a simple yet effective weighted quantization loss function to balance the importance of tokens. Furthermore, by alternately employing orthogonal Procrustes transformations to refine the rotation matrix R1 and optimizing quantization parameters for X, our method, named DFRot, enhances the Rotated LLMs by achieving Dual Free, including *Outlier-Free* and *Massive Activation-Free*. It is worth noting that DFRot significantly improves model accuracy in 4-bit activation quantization with just a single data sample, achieving PPL improvements of 0.98 and 0.95 on W4A4KV4 and W4A4KV16, respectively, for the LLaMA3-70B, which is notable for its quantization challenge.

### References
*(参考文献列表较长，为简洁起见此处省略，但内容与原文一致)*

---
### A Calibration Data

In this section, we explain the reason why we only used a single data sample to calibrate the rotation matrix R₁ in DFRot, and don not attempt to use more data:

- In LLMs, outliers and massive activations often appear in some fixed channels. Therefore, the process of optimizing the rotation matrix can be seen as an optimization of the distribution patterns of outliers and massive activations. We have simply use ten samples to calibrate the rotation matrix for LLaMA2-7B, but no significant improvement in accuracy was observed.
- Our calibration data is a sample with a length of 2048 tokens. Since we obtain the calibration set from each MHA and FFN, taking LLaMA2-7B as an example, we can obtain 2048 × 32 × 2 = 131072 tokens as calibration tokens. This is relatively sufficient to statistically analyze the distribution patterns of outliers and massive activations.

### B Limitations

Due to the discontinuity of the quantization function, our current optimization method is prone to getting stuck in local minima and often relies on initialization. As can be seen from Figure 5 and Figure 6, the performance of RH is better than that of RO in most cases. Meanwhile, during the optimization of the rotation matrix, we also found that after iterative convergence, the final quantization error optimized with RH initialization is often better than that with RO.

### C Algorithm

**Algorithm 1 Optimization of Quantization Parameters and Rotation Matrix**
**Require:** Token x, initial rotation matrix R₁, quantization function Q
**Ensure:** Optimized rotation matrix R₁ and quantization parameters ηₓ

1: Initialize R₁ with randomized Hadamard matrix, t = 0
2: **while** t ≤ 100 **do**
3:    // Step 1: Optimize Quantization Parameters ηₓ
4:    **for** each token x **do**
5:       Compute quantization parameters s, z via arg min\_{s,z} ||xR₁ᵗ⁻¹ – Q(xR₁ᵗ⁻¹, sₓ, zₓ) ||²
6:       Update ηₓᵗ = Q(xR₁ᵗ⁻¹, sₓᵗ, zₓᵗ)
7:    **end for**
8:    // Step 2: Optimize Rotation Matrix R₁
9:    Solve the Procrustes problem to update R₁ᵗ: R₁ᵗ = arg min\_{R} ||XR – ηₓᵗ||²
10:   t = t + 1
11: **end while**
12: **return** Optimized R₁ᵗ

### D Quantization error for tokens with Massive activation in LLaMA-7B, LLaMA2-7B, LLaMA2-13B

More quantization results for LLaMA-7B, LLaMA2-7B and LLaMA2-13B:

**Figure 7: Comparison of 4-bit quantization error for the token with massive activation with NR, RO, RH and DFRot for LLaMA-7B from Figure 2.**
*(图表内容为散点图，此处省略)*

**Figure 8: Comparison of 4-bit quantization error for the token with massive activation with NR, RO, RH and DFRot for LLaMA2-7B from Figure 2.**
*(图表内容为散点图，此处省略)*

**Figure 9: Comparison of 4-bit quantization error for the token with massive activation with NR, RO, RH and DFRot for LLaMA2-13B from Figure 2.**
*(图表内容为散点图，此处省略)*

### E Quantization error between NR, RO, RH and DFRot

More 2D quantization error visualization are shown as follows:

**Figure 10: Comparison of 4-bit quantization error for the token with massive activation with NR, RO, RH and DFRot for LLaMA-7B from Figure 2.**
*(图表内容为量化误差图，此处省略)*

**Figure 11: Comparison of 4-bit quantization error for the token with massive activation with NR, RO, RH and DFRot for LLaMA2-7B from Figure 2.**
*(图表内容为量化误差图，此处省略)*

**Figure 12: Comparison of 4-bit quantization error for the token with massive activation with NR, RO, RH and DFRot for LLaMA2-13B from Figure 2.**
*(图表内容为量化误差图，此处省略)*

### F Visualization for Different layers

We visualize for more layer as follwing:

*(此部分包含 Figure 13 至 Figure 24，均为不同层激活的可视化3D散点图，此处省略图表)*

### G Full Results

**Table 5: Complete omparison of the perplexity score on WikiText2 and averaged accuracy on Zero-shot Common Sense Reasoning tasks on LLaMA-2 & 3.**

| Model | #Bits W-A-KV | Method | ARC-c (↑) | ARC-e (↑) | BoolQ (↑) | Hellas. (↑) | Lam. (↑) | OBQA (↑) | PIQA (↑) | SIQA (↑) | WinoG. (↑) | Avg. (↑) | Wiki2 (↓) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2-7B** | 16-16-16 | Full Precision | 46.42 | 74.33 | 77.71 | 75.94 | 73.69 | 44.20 | 79.16 | 45.91 | 69.53 | 65.21 | 5.47 |
| | 4-4-16 | RTN | 25.34 | 28.03 | 50.52 | 27.71 | 1.01 | 26.20 | 50.82 | 33.93 | 48.38 | 32.44 | nan |
| | | SmoothQuant | 28.33 | 26.39 | 49.39 | 27.28 | 1.18 | 23.40 | 48.80 | 33.62 | 50.75 | 32.13 | nan |
| | | GPTQ | 24.40 | 28.70 | 51.62 | 28.66 | 1.36 | 24.60 | 51.14 | 34.49 | 49.49 | 32.72 | nan |
| | | QuaRot | 45.99 | 72.73 | 77.8 | 75.92 | 73.45 | 43.8 | 78.02 | 46.16 | 68.03 | 64.66 | 5.45 |
| | | DFRot | 44.97 | 70.24 | 74.34 | 73.23 | 68.99 | 42.60 | 76.82 | 44.06 | 66.54 | 62.42 | 6.13 |
| | | SpinQuant* | 37.54 | 62.58 | 71.16 | 70.48 | 67.16 | 34.80 | 75.46 | 39.76 | 60.62 | 57.37 | 6.78 |
| | | OSTQuant* | 44.03 | 71.93 | 75.41 | 74.94 | 73.22 | 43.20 | 78.51 | 45.85 | 68.03 | 63.90 | 5.60 |
| | 4-4-4 | RTN | 27.22 | 27.06 | 50.83 | 27.34 | 0.93 | 25.80 | 49.51 | 34.85 | 50.51 | 32.67 | nan |
| | | SmoothQuant | 26.37 | 25.63 | 47.71 | 27.05 | 1.11 | 26.40 | 51.90 | 34.49 | 48.38 | 32.12 | nan |
| | | GPTQ | 26.96 | 27.65 | 52.84 | 28.83 | 1.63 | 29.20 | 49.62 | 35.11 | 49.80 | 33.52 | nan |
| | | Omniquant | 31.40 | 53.75 | 63.79 | 55.06 | 35.63 | 34.40 | 66.59 | 40.28 | 54.70 | 48.40 | 14.26 |
| | | QuaRot | 41.3 | 69.32 | 72.66 | 72.09 | 69.67 | 39.0 | 76.5 | 43.19 | 63.54 | 60.81 | 6.25 |
| | | DFRot | 43.52 | 70.83 | 73.30 | 72.62 | 68.46 | 41.40 | 76.82 | 44.17 | 65.11 | 61.80 | 6.25 |
| | | SpinQuant* | 40.44 | 71.08 | 74.40 | 73.51 | 70.66 | 41.80 | 76.88 | 43.50 | 65.82 | 62.01 | 5.96 |
| | | OSTQuant* | 42.92 | 72.56 | 74.71 | 73.14 | 71.76 | 44.40 | 77.42 | 44.98 | 66.77 | 63.18 | 5.91 |
| **2-13B** | 16-16-16 | Full Precision | 49.15 | 77.53 | 80.58 | 79.39 | 76.62 | 45.20 | 80.63 | 47.49 | 71.90 | 67.61 | 4.88 |
| | 4-4-16 | RTN | 27.99 | 26.81 | 38.50 | 26.08 | 0.00 | 23.60 | 48.20 | 34.90 | 51.62 | 30.86 | 8e3 |
| | | SmoothQuant | 24.49 | 35.06 | 47.98 | 30.87 | 3.67 | 26.20 | 55.01 | 35.31 | 49.72 | 34.26 | 1e3 |
| | | GPTQ | 27.82 | 26.77 | 37.92 | 25.67 | 0.00 | 21.80 | 47.77 | 35.11 | 48.15 | 30.11 | 4e3 |
| | | QuaRot | 46.42 | 73.86 | 78.10 | 75.68 | 74.31 | 43.00 | 79.05 | 44.37 | 71.35 | 65.13 | 5.35 |
| | | DFRot | 46.67 | 74.45 | 77.19 | 77.07 | 75.04 | 43.6 | 78.29 | 46.16 | 69.61 | 65.34 | 5.39 |
| | | SpinQuant* | 43.77 | 69.99 | 76.57 | 74.63 | 72.81 | 41.60 | 77.20 | 44.27 | 68.19 | 63.23 | 5.24 |
| | | OSTQuant* | 47.78 | 74.66 | 80.03 | 77.60 | 75.94 | 44.40 | 79.38 | 46.06 | 70.32 | 66.24 | 5.14 |
| | 4-4-4 | RTN | 27.82 | 26.52 | 38.38 | 26.27 | 0.02 | 26.00 | 49.78 | 34.39 | 49.17 | 30.93 | 7e3 |
| | | SmoothQuant | 24.49 | 33.00 | 45.84 | 30.70 | 2.70 | 23.80 | 53.81 | 34.80 | 51.07 | 33.36 | 2e3 |
| | | GPTQ | 27.90 | 26.39 | 37.95 | 26.16 | 0.00 | 27.00 | 48.26 | 34.39 | 50.43 | 27.85 | 5e3 |
| | | Omniquant | 32.85 | 55.13 | 64.34 | 60.13 | 42.85 | 33.40 | 68.17 | 39.76 | 56.51 | 50.35 | 12.30 |
| | | QuaRot | 44.37 | 73.32 | 77.58 | 75.73 | 73.16 | 43.0 | 78.84 | 45.04 | 68.90 | 64.44 | 5.49 |
| | | DFRot | 46.50 | 73.48 | 76.67 | 76.83 | 73.92 | 43.00 | 79.27 | 45.55 | 69.30 | 64.95 | 5.43 |
| | | SpinQuant* | 46.67 | 74.49 | 76.76 | 75.22 | 72.19 | 42.40 | 78.29 | 43.45 | 67.72 | 64.13 | 5.74 |
| | | OSTQuant* | 47.10 | 75.21 | 77.46 | 76.71 | 75.14 | 44.60 | 78.67 | 45.75 | 68.03 | 65.41 | 5.25 |
| **3-8B** | 16-16-16 | Full Precision | 53.50 | 77.74 | 81.10 | 79.18 | 75.74 | 44.80 | 80.63 | 47.08 | 73.01 | 68.09 | 6.14 |
| | 4-4-16 | RTN | 23.72 | 30.89 | 46.30 | 29.19 | 1.57 | 28.60 | 52.72 | 35.26 | 50.04 | 33.42 | 6e2 |
| | | SmoothQuant | 23.29 | 28.28 | 48.93 | 31.26 | 2.70 | 28.60 | 54.46 | 33.37 | 49.64 | 33.04 | 1e3 |
| | | GPTQ | 23.46 | 32.07 | 43.79 | 30.10 | 2.41 | 28.00 | 53.97 | 34.14 | 48.86 | 32.98 | 6e2 |
| | | QuaRot | 44.03 | 69.74 | 71.90 | 73.16 | 67.26 | 42.4 | 76.71 | 45.04 | 66.46 | 61.86 | 8.11 |
| | | DFRot | 46.16 | 72.90 | 73.73 | 74.98 | 68.23 | 40.60 | 77.53 | 44.32 | 68.67 | 63.01 | 7.78 |
| | | SpinQuant* | 47.35 | 74.12 | 76.36 | 75.98 | 69.88 | 42.46 | 77.37 | 44.47 | 68.98 | 64.11 | 7.28 |
| | | OSTQuant* | 48.81 | 73.48 | 79.82 | 75.97 | 72.62 | 42.40 | 78.18 | 45.75 | 69.22 | 65.14 | 7.24 |
| | 4-4-4 | RTN | 23.72 | 30.56 | 46.18 | 29.83 | 2.72 | 27.60 | 52.45 | 34.39 | 50.20 | 33.18 | 7e2 |
| | | SmoothQuant | 23.55 | 28.96 | 48.84 | 28.90 | 1.44 | 29.40 | 51.09 | 34.14 | 50.36 | 32.96 | 1e3 |
| | | GPTQ | 23.38 | 32.74 | 44.34 | 29.72 | 2.39 | 29.80 | 54.95 | 34.75 | 51.30 | 33.71 | 6e2 |
| | | Omniquant | 22.87 | 30.35 | 41.53 | 31.11 | 1.86 | 25.40 | 53.37 | 34.08 | 50.43 | 32.33 | 4e2 |
| | | QuaRot | 43.17 | 70.58 | 72.66 | 72.53 | 66.66 | 39.20 | 76.06 | 44.83 | 66.77 | 61.38 | 8.28 |
| | | DFRot | 44.97 | 71.09 | 73.27 | 74.13 | 67.63 | 43.00 | 78.24 | 44.58 | 69.53 | 62.94 | 7.91 |
| | | SpinQuant* | 46.33 | 73.57 | 76.15 | 75.43 | 71.40 | 41.40 | 79.16 | 44.68 | 68.75 | 64.10 | 7.35 |
| | | OSTQuant* | 49.32 | 76.73 | 78.87 | 76.01 | 70.77 | 43.20 | 78.51 | 45.70 | 69.22 | 65.37 | 7.29 |

