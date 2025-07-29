# Document 1: QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

**arXiv:2404.00456v2 [cs.LG] 29 Oct 2024**

## QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

**Saleh Ashkboos**
ETH Zurich
saleh.ashkboos@inf.ethz.ch

**Amirkeivan Mohtashami**
EPFL
amirkeivan.mohtashami@epfl.ch

**Maximilian L. Croci**
Microsoft Research
mcroci@microsoft.com

**Bo Li**
ETH Zurich
bolibo@ethz.ch

**Pashmina Cameron**
Microsoft
pcameron@microsoft.com

**Martin Jaggi**
EPFL
martin.jaggi@epfl.ch

**Dan Alistarh**
IST Austria & NeuralMagic
dan.alistarh@ist.ac.at

**Torsten Hoefler**
ETH Zurich
torsten.hoefler@inf.ethz.ch

**James Hensman**
Microsoft Research
jameshensman@microsoft.com

### Abstract

We introduce QuaRot, a new Quantization scheme based on Rotations, which is able to quantize LLMs end-to-end, including all weights, activations, and KV cache in 4 bits. QuaRot rotates LLMs in a way that removes outliers from the hidden state without changing the output, making quantization easier. This computational invariance is applied to the hidden state (residual) of the LLM, as well as to the activations of the feed-forward components, aspects of the attention mechanism, and to the KV cache. The result is a quantized model where all matrix multiplications are performed in 4 bits, without any channels identified for retention in higher precision. Our 4-bit quantized LLAMA2-70B model has losses of at most 0.47 WikiText-2 perplexity and retains 99% of the zero-shot performance. We also show that QuaRot can provide lossless 6 and 8 bit LLAMA-2 models without any calibration data using round-to-nearest quantization. Code is available at github.com/spcl/QuaRot.

### 1 Introduction

Large language models (LLMs) have become increasingly important due to their countless applications. However, using these models in practice, known as inference, requires a significant amount of computation, memory, and energy, specifically during the *prefill* phase, in which the model is supposed to process large prompts and cache them in each layer. Quantization is among the most important techniques to improve both memory and compute issues by keeping the data types at lower precision during the forward pass.

As the prefill stage is known to be compute-bound [Ashkboos et al., 2023], joint quantization aims to reduce the precision of parameters and KV cache (which results in lower memory usage) as well as inputs (known as activations) and compute the forward pass in low precision. However, quantizing the activations is hard as they have large outlier elements (see Figure 1 for an illustrative example) with much larger values, making activation quantization more difficult than weight quantization, especially for the 4-bit case. Previous work relies on using a calibration set to characterize the outlier features and keeping them in higher precision for inference [Zhao et al., 2023, Ashkboos et al., 2023].

*38th Conference on Neural Information Processing Systems (NeurIPS 2024).*

---



**Figure 1: The distributions of activations at the input to the FFN block in LLAMA2-7B model, in the tenth layer. Left: using the default configuration as downloaded from Hugging Face. Right: after processing using QuaRot. The processed distribution has no outliers, leading to superior quantization.**

In this work, we address the issue of outlier features by rotating the inputs of the model using randomized Hadamard transformations. We do this using the *computational invariance* idea [Ashkboos et al., 2024] and fuse Hadamard transformations into the weight matrices, resulting in an equivalent network without outlier features. This enables the weights, activations, and KV caches to be quantized to 4 bits with minimal accuracy drop. Our main contributions are:

- We show that randomized Hadamard transformations can be applied to the weight matrices without additional model modifications. In turn, this completely eliminates outlier features and makes the activations easy to quantize, without changing the output of the model. This can be seen as an extension of the computational invariance idea, proposed in SliceGPT [Ashkboos et al., 2024] in the context of structured pruning.
- We extend this approach to apply *online* Hadamard transformations to the attention module to remove outlier features in keys and values, enabling the KV cache to be quantized.
- Using the above modifications, QuaRot enables 4-bit LLM inference by quantizing all weights, activations, and KV caches using integer quantization. We provide efficient kernel support for QuaRot: on a LLAMA2-70B model, QuaRot achieves up to 3.33× prefill speedups (on a batch size 64 with 2048 sequence length), and 3.89× memory saving during the decoding stage, with at most 0.47 WikiText-2 perplexity loss. QuaRot preserves 99% of the accuracy of zero-shot tasks and we show that our 6 and 8-bit quantization is lossless with simple round-to-nearest quantization.

### 2 Related Work

The majority of quantization schemes focus on compressing LLMs by using *weight-only quantization*, [Frantar et al., 2022, Dettmers et al., 2023, Lin et al., 2023, Egiazarian et al., 2024, Tseng et al., 2024]. These methods downcast each weight into a low-precision representation and upcast it before the actual computation. The main computation is still performed in high precision. Several works show that, unlike weights, quantizing the activations is hard due to the outlier features [Wei et al., 2022, Dettmers et al., 2022, Xiao et al., 2023]. For 8-bit case, LLM.int8() [Dettmers et al., 2022] identifies the outlier features during inference and keeps them in 16 bits which results in poor performance. SmoothQuant [Xiao et al., 2023] normalizes the features using some scaling factors from a calibration set, solving the issue for the 8-bit case at the cost of introducing extra hyper-parameters. For 4-bit quantization, recent studies identify the outlier features offline and keep them in high precision. Atom [Zhao et al., 2023] developed a complex kernel for mixed-precision MatMul in the presence of outliers while QUIK [Ashkboos et al., 2023] keeps the down-projection layer in 8 bits.

Two weight-only quantization methods, QuIP [Chee et al., 2024] and QuIP# [Tseng et al., 2024] have previously considered improving quantization by applying rotations. Chee et al. [2024] introduced the idea of *incoherence processing* which applies rotation matrices to the left and right of each weight matrix, as well as the Hessian, which is used in minimizing the weight-quantization objective. Xi et al. [2023] uses a similar idea during training, using exact Hadamard transformations for each linear layer in the forward pass.

Finally, KV cache quantization is another line of research that aims to compress the cached keys and values during the generation phase. This is crucial for large batch size and long-context length generation as the KV cache will be the main memory bottleneck in such problems. Sheng et al. [2023] quantizes the KV cache using 4-bit group-wise quantization. KVQuant [Hooper et al., 2024] pushes this limit to 3-bit quantization and KIVI [Liu et al., 2024] shows promising results on 2-bit KV cache quantization. Such methods show that outliers also exist in the keys, and apply a set of complex ideas (like feature-wise quantization, non-uniform representation, and keeping high precision outliers) to recover the accuracy of a quantized KV cache.

In this work we also adopt the Hadamard transform to improve quantization of weights through incoherence processing. Instead of undoing the Hadamard transform during the forward pass, we adopt the *computational invariance* theorem from SliceGPT [Ashkboos et al., 2024] to fuse the transformations into the weights where possible. Instead of requiring two Hadamard transforms per weight-matrix in the forward pass, QuaRot requires just 1½ Hadamard transforms per transformer layer. Computational invariance also means that the *activations* are incoherence-processed, enabling them to be effectively quantized. We also apply a similar technique to the attention block and quantize the KV cache in 4 bits with minimal accuracy loss.

### 3 Background

Here we introduce some mathematical concepts and notation that are necessary for QuaRot.

#### 3.1 Orthogonal, Rotation and Hadamard Matrices

An orthogonal matrix Q is a square matrix such that QQT = I. In this work, we consider only real orthogonal matrices. A rotation matrix is an orthogonal matrix. A Hadamard matrix is an orthogonal matrix with entries drawing from {+1,-1}. A Walsh-Hadamard matrix is a square matrix of size d = 2n, with

H₂ = (1/√2) * [[1, 1], [1, -1]] and H₂ₙ = H₂ ⊗ H₂ₙ₋₁. (1)

These identities give rise to the Walsh-Hadamard transform, which computes the matrix-vector product Hx in O(dlog₂(d)) operations.

For matrix sizes that are not 2n, the existence of a Hadamard matrix is not guaranteed. A useful list of known Hadamard matrices is made available by Sloane [2024]. Where we require a Hadamard matrix of size d≠ 2n, we factorize d = 2nm, where m is the size of a known Hadamard matrix. Then we use a Kronecker construction H_d = H₂ₙ ⊗ Hₘ. This allows computation of H_dx in O(d(m + n)) operations.

Following Tseng et al. [2024] we make use of *randomized* Hadamard matrices where convenient. Let s be a vector containing random draws from {+1,−1}, and Ĥ = H diag(s). It is straightforward to see that Ĥ is also an orthogonal matrix.

#### 3.2 Incoherence Processing

The idea of *incoherence processing* was introduced by [Chee et al., 2024] in the context of weight normalization for weight-only LLM quantization. We define a weight matrix W to be µ-incoherent if

max(W) ≤ µ||W||_F/√mn (2)

where max is the element-wise max of the matrix, and mn is the number of elements. A weight matrix that has high incoherence is hard to quantize: the largest element is an outlier relative to the magnitude of the average element. Chee et al. [2024] showed that multiplying a weight matrix on the left and right by an orthogonal matrix can reduce the incoherence, making matrices easier to quantize. In this work we adopt a similar technique, multiplying weight matrices by orthogonal matrices to improve incoherence, though we add fewer operations to the forward pass. Importantly, we additionally apply incoherence processing to the activations, enabling improved weight and activation quantization. Figure 1 shows the effect of applying incoherence processing to the activations of LLAMA-2.

---



**Figure 2: The gated feed-forward network used in most LMs, including the pre-positioned RMSNorm. The input signal is divided by its norm, and re-scaled by parameters α. Two linear blocks, W_up and W_gate are applied. The activation function σ is applied to the gated signal, and the two signals are element-wise multiplied together. The final linear block W_down produces the output signal Y. Before quantization, different operations are performed either in single (32 bit) or half (16 bit) precision.**

#### 3.3 Transformer structures

Large Language Models are neural networks with repeating attention and feed-forward layers. We introduce our notation through Figures 2 and 5, which show the construction of these blocks. We assume that the construction of the network is “pre-norm", in that each block is preceded by a LayerNorm or RMSNorm operation. We also assume that the feed-forward network uses a gated architecture, as in LLAMA-2, though our methodology is straightforwardly applied to MLP architectures also.

#### 3.4 Computational Invariance

The computational invariance theorem [Ashkboos et al., 2024, Theorem 1] states that the weights and between-block activations in a transformer can be transformed using an orthogonal matrix with no change to the model output. Here we sketch the main idea. If W_in is a weight matrix that appears on the left of a transformer block (i.e., W_gate, W_up in Figure 2, or W_k, W_q, W_v in Figure 5) then we can multiply on the left by an orthogonal matrix Q, and cancel out this effect by multiplying the output matrix (W_down, W_out) by Qᵀ. This applies despite the fact that RMSNorm is applied between the two blocks, so long as no re-scaling happens in the RMSNorm block (and in practice, we absorb any re-scaling into adjacent weight matrices first). Conceptually, this is because RMSNorm divides the activations by their norm, and applying a rotation Q to the activations does not affect the norm. We have the commutation property

RMSNorm(X) = RMSNorm(XQᵀ)Q, (3)

where we assume here that RMSNorm applied to each row of the activations X as xᵢ ← xᵢ/||xᵢ||. This means that multiplying an output matrix by Qᵀ makes the linear layer output XQᵀ, which is normalized and then passed into the next block whose input weight matrix is now QW, and so this linear layer outputs the original activations without modification.

### 4 Method

QuaRot consists of two stages. In the first stage, the model weights are manipulated (in full precision), and two additional Hadamard operations are inserted into the model's forward pass. In the second stage, the weights are quantized using some existing method, and quantization operations are added to the forward pass to enable on-line quantization of the activations (and caches). By default, we use GPTQ [Frantar et al., 2022] for quantizing weights, whilst activations are quantized on-the-fly using a simple round-to-nearest scheme. Figures 3 and 6 show updated block diagrams for the forward pass with QuaRot modifications, including updated weight matrices, inserted blocks and the bit-width of weights and activations.

**Stage 1a: Weight Modification.** We first make use of computational invariance to multiply each weight matrix by an orthogonal matrix. To enable this, the linear parts of LayerNorm or RMSNorm are fused into adjacent weight matrices. Figure 3 shows how the feed-forward block of a transformer is modified by removing the scaling operation from RMSNorm (diag(α)) and absorbing into the subsequent weight matrices. We select a randomized Hadamard matrix with size that matches the hidden dimension of the model and pre- or post-multiply each weight matrix. In Figures 3 and 6 this matrix is denoted Q. For example the key-projection weight matrix W_k is modified as

W_k ← Qᵀdiag(α)W_k, (4)

and similarly for other weight matrices. Matrices that appear on the output side of a block are post-multiplied by Q.

This weight modification does not affect the output of the model (assuming sufficient precision) as per the computational invariance theorem [Ashkboos et al., 2024]. We note that the modified weights resemble the modifications used in QuIP# [Tseng et al., 2024], reducing the incoherence of the weights, though our modification does not require any additional processing at run-time. Additionally, the activation matrix passed between blocks of the transformer is also incoherence processed, becoming X ← XQ. Figure 1 shows the result of this processing: we see that the processed activations no longer contain any outliers.

**Stage 1b: Rotate FFN activations.** With the above weight-modifications in place, we have multiplied many weight matrices on one side by a Hadamard matrix and the activations have been changed. It remains to improve the quantization of the activations *within* each block, which we achieve by inserting on-line Hadamard operations.

We first insert a Hadamard operation into the feed-forward network, before the down-projection matrix. This operation is performed in full precision, and implemented using a fast kernel following Tseng et al. [2024]. This operation is implicitly reversed by fusing a Hadamard matrix into the down-projection matrix of the network: W_down ← HW_down. Combined with the global matrix Q, this means that the down-projection matrix now becomes HW_downQ (see Figure 3).

---



**Figure 3: QuaRot applied to a LLaMa-style FFN. The RMSNorm scaling (α) has been absorbed into the weight matrices ((α) is a diagonal matrix with RMSNorm parameters). The hidden state X has been rotated by Q, which is canceled out by the absorption of Qᵀ into the first two weight matrices. All weights are stored in INT4, and all activations immediately before the weights are also quantized to INT4. The result of the matmul between the INT4 weights and activations on a TensorCore is INT32, which we immediately cast (and scale) to FP16 which is the default precision of the model. Whilst the signal is still in FP16, we perform a single on-the-fly Hadamard transform before quantizing and computing a (modified) down-proj, which results in a rotated output YQ.**

---

**Stage 1c: Attention Value Projection.** Next, we apply an additional Hadamard operation to each attention block. This modification is partially on-line, and partially fused into the weight matrices as we will now detail.

First, note that in the computation of attention, the W_v and W_out matrices are implicitly multiplied together within each head. To see this, note that the attention computation consists of

Y = concat[(P₁V₁) ... (P_nh V_nh)] W_out (5)

= Σ (from h=1 to H) P_h X W_v^(h) W_out^(h) (6)

where P_h is a sequence-length sized square matrix computed by softmaxing keys and values, and V_h = XW_v^(h) is the value matrix for one head. This presents an opportunity to perform additional processing on W_v and W_out using a Hadamard matrix H_dh, which matches the dimension of each head:

W_v^(h) ← W_v^(h)H_dh , W_out^(h) ← H_dh W_out^(h). (7)

Substituting these modifications into equation (6), we see that the computed result of attention remains unchanged. Since the weights for each head are concatenated in the weight representation, we can equivalently perform a single Kronecker structured multiplication:

W_v ← W_v(I ⊗ H_dh), W_out ← (I ⊗ H_dh)W_out (8)

This transformation has now been applied head-wise to the weight matrices, and results in computed activations (emitted by the block multi-head attention) rotated head-wise also. To complete a "full" Hadamard operation on the attention-activations, sharing the transform across heads, we make use of the identity

H_nh x d_h = (I ⊗ H_dh)(H_nh ⊗ I) (9)

which holds when the number of heads n_h and the dimension of each head d_h are both powers of 2. Since we have already applied (I ⊗ H_dh) to both W_v and W_out, it remains to apply (H_nh ⊗ I) to W_out, which results in a complete transformation of W_out ← HW_out, and to insert a block into the forward pass that computes Z ← Z(H_nh ⊗ I) where Z is the attention activation. This block is denoted *Hadamard heads* in Figure 6 and can be computed efficiently using a reshape to deal with the Kronecker structure, and a Walsh-Hadamard transform on the reshaped data.

**Stage 1d: Key Rotation.** Using the method above, we can successfully quantize the value vectors. However, key vectors in the attention module are also known to suffer from outliers [Hooper et al., 2024, Liu et al., 2024]. Similar to above, we can use a Hadamard rotation to alleviate this issue, allowing us to have a fully quantized KV cache. First note that the attention scores P₁, . . ., P_h are computed as:

Q ← Pos(XW_q) = concat[Pos(Q₁), ..., Pos(Q_nh)] (10)
K ← Pos(XW_k) = concat[Pos(K₁), ..., Pos(K_nh)] (11)
P_h ← Softmax(α Pos(Q_h) Pos(K_h)ᵀ + M), (12)

where α is the Softmax scale usually set to 1/√d_k, M is the attention mask (e.g., causal), and Pos denotes the positional embedding. Previously, positional embedding was only added before the first layer to the input, in which case Pos is an identity function. However, recent methods such as RoPE [Su et al., 2021] add position information directly to the key and query vectors.

We can now observe the same interaction between Q and K as we observed between W_v and W_out. However, the existence of Pos prevents us from directly fusing the Hadamard matrix into W_q and W_k. Therefore, we use online head-wise Hadamard rotation to rotate both the queries and keys. As a result, the computation of query and key matrices is altered as follows:

Q ← Pos(XW_q)(I ⊗ H_dh) = concat[Pos(Q₁)H_dh, ..., Pos(Q_nh)H_dh] (13)
K ← Pos(XW_k)(I ⊗ H_dh) = concat[Pos(K₁)H_dh, ..., Pos(K_nh)H_dh]. (14)

Since both queries and keys are rotated, the final attention scores P₁, ..., P_h remain unchanged. We note that an alternative to the above process is caching the keys before applying the positional encoding. This approach (called Pre-RoPE Caching [Hooper et al., 2024]) needs the inverse rotation to be applied online before applying the positional encoding but removes the need to rotate the query vector. It also adds the overhead of rotating the keys and values for every query. Given that at the time of decoding there is a single query vector and many cached key vectors, we use Post-RoPE caching. This helps us to apply a Hadamard transformation on a single token at each decoding step.

Overall, our modifications to the forward pass, including the insertion of special Hadamard blocks and adjustments to the weights do not change the forward pass of the model. The effect is that the activations between blocks have been multiplied by a Hadamard matrix, and the activations within blocks are processed on-line using Hadamard transforms in a way that is undone by corresponding weight matrix modifications. We are now ready to quantize the weights and activations.

**Stage 2a: Weight Quantization.** We apply GPTQ [Frantar et al., 2022] to quantize the weights of the network. We note that after the above forward-pass modifications, any quantization method could be applied. In subsequent sections, we show that a simple round-to-nearest (RTN) scheme can be applied instead of GPTQ, at the cost of some accuracy.

**Stage 2b: Online Quantization Operations.** With the weights quantized, we are ready to apply operations to the forward pass that quantize the activations. Following PyTorch implementation, we leave the computation of RMSNorm (without scaling) in FP32. We quantize the input of the linear layers using symmetric per-token (rows of the input matrix). During symmetric quantization, the row scales are computed by dividing the maximum absolute value of each token by 7 (largest representable number in INT4). We then divide each row to its corresponding scale and round the result to its nearest integer. The dequantization is also done by casting the INT32 output of GEMM into FP16, multiply the corresponding scale for the row (from input scales) and column (from weight scales).

**Stage 2c: Quantized Attention.** Attention is significantly memory bound for longer sequences and larger batch sizes. Having rotated both keys and values, we can successfully quantize the cache into low bit-width. This reduces the number of IO operations needed. We keep the queries in FP16 and use online softmax calculation similar to Flash Attention [Dao et al., 2022]. After a segment of the KV vectors are loaded from the memory, we dequantize and compute the dot product in FP16.

### 5 Experimental Validation

**Setup.** We implement QuaRot using Hugging Face [Wolf et al., 2019] on top of the PyTorch framework [Paszke et al., 2019]. To quantize the inputs, we use per-token symmetric quantization (a single scale for every row) with a constant clipping ratio of 0.9 in all our experiments. We quantize the KV caches using asymmetric quantization with a group size 128 with a constant clipping ratio of 0.95. For weight quantization, we use round-to-nearest (RTN) and GPTQ [Frantar et al., 2022] with per-column (also known as per-channel) symmetric quantization, where we extract the clipping ratio using a linear search over the squared error. We use 128 samples from WikiText-2 [Merity et al., 2016] training set with 2048 sequence length as the calibration set during GPTQ quantization. On a single NVIDIA A100 GPU, modifying LLAMA2-70B with QuaRot takes 5 minutes and quantizing the model with GPTQ takes a further 2 hours. We present LLAMA-3 results in Appendix A.8.

**Models, Tasks, and GPUs.** We evaluate QuaRot on the LLAMA-2 family [Touvron et al., 2023] on both language generation and zero-shot tasks. We implement our low-level CUDA kernel to perform 4-bit matrix-multiplication using the CUTLASS [NVIDIA, 2023] library. We use the FlashInfer [Ye, 2023] library for implementing our KV cache quantization. As we target consumer-type GPUs, we evaluate all the performance experiments on NVIDIA RTX 3090 GPUs.

#### 5.1 Accuracy Results

**Language Generation Tasks.** First, we evaluate the accuracy of QuaRot on the language generation task. Table 1 shows the perplexity of LLAMA-2 models on WikiText-2 when we quantize the weights using GPTQ. We compare against 4-bit SmoothQuant [Xiao et al., 2023] and OmniQuant [Shao et al., 2023]. We also include the QUIK [Ashkboos et al., 2023] results when they keep all the layers (including down-projection) in 4 bits. QuaRot outperforms all previous work with at most 0.63 perplexity loss (0.47 on LLAMA2-70B model) without any re-training (as in OmniQuant) nor higher precision outlier features and asymmetric quantization (as in QUIK). We also apply group-wise quantization to compare against Atom [Zhao et al., 2023] on the same number of groups for weight and activations. In this setting, QuaRot doesn't need to keep any higher precision features and related operations (like re-ordering). QuaRot outperforms Atom with 0.1 perplexity points in the 7B model. On the 13B model, we get the same perplexity number as Atom.

**Zero-Shot Tasks.** Next, we focus on evaluating QuaRot on six important zero-shot tasks: PIQA [Bisk et al., 2020], WinoGrande [Sakaguchi et al., 2021], HellaSwag [Zellers et al., 2019], LAMBADA (OpenAI) [Radford et al., 2019], and Arc (Easy and Challenge) [Clark et al., 2018]. We use the LM Evaluation Harness [Gao et al., 2021] with default parameters for our experiments. Table 2 shows the accuracy of our scheme on the above tasks as well as the average score. On LLAMA-2 family, QuaRot preserves the accuracy with at most 4.18% average score loss (1.09% for 70B model).

#### 5.2 Performance Analysis

We implement QuaRot using CUDA/12.1 on top of PyTorch and use CUTLASS for performing INT-4 matrix multiplication on TensorCore (where the results will be saved in an INT32 accumulator). In this section, we evaluate the performance of our kernels for both prefill and decoding steps on NVIDIA RTX 3090 GPU. We provide all our experiments on a single transformer block as the whole model does not fit on our GPU cluster for large batch sizes. We provide more performance analysis of our kernels (as well as complete results) in Appendix A.10.

**Prefill Stage Performance Increases.** For the compute-bound prefill stage, we present the speedups of using QuaRot on 2048 sequence length with different batch sizes in Figure 4 Left. On LLAMA2-7B model, we get 1.97x-2.16x speedup over the FP16 implementation using our QuaRot kernel. The speedup increases with batch sizes as the computation will become a bottleneck in larger batch sizes. on LLAMA2-70B model, we get up to 3.33x speedup. Note that our performance results could be improved by optimizing our kernels (e.g., fusing the quantization operations into the MatMul).

**Decoding Stages Memory Saving.** Finally, we evaluate the memory improvement which is the main bottleneck of the decoding stage. Figure 4 Right shows the peak memory saving on LLAMA-2 models. We provide results for LLAMA2-7B and LLAMA2-70B models. In both models, we get at least 3.63x peak memory saving compared to FP16 case during the decoding stage. Note that the KV cache is larger in LLAMA2-7B model as the LLAMA2-70B uses grouped-query attention [Ainslie et al., 2023]. In the LLAMA2-7B model, the memory saving increases with the sequence length, resulting in up to 3.75x memory saving. on LLAMA2-70B model, we get 3.89x savings in almost all the cases. We expect these values to be larger for the whole model (instead of just the single layer here) since as the number of layers increases the effect of constant size objects in memory becomes much less significant.

---
**Table 1: WikiText-2 perplexity results on 4-bit quantization of LLAMA-2 models with 2048 sequence length. We extract the results for SmoothQuant and OmniQuant results of [Shao et al., 2023]. 128G shows the group-wise quantization with group size 128.Here, we quantize all weights, activations, and caches in 4-bits in QuaRot.**

| Method | Weight Quantization | #Outlier Features | 7B | 13B | 70B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | - | - | **5.47** | **4.88** | **3.32** |
| SmoothQuant | RTN | 0 | 83.12 | 35.88 | - |
| OmniQuant | RTN | 0 | 14.26 | 12.30 | - |
| QUIK-4B | GPTQ | 256 | 8.87 | 7.78 | 6.91 |
| **QuaRot** | **GPTQ** | **0** | **6.10** | **5.40** | **3.79** |
| Atom-128G | GPTQ-128G | 128 | 6.03 | 5.26 | - |
| **QuaRot-128G** | **GPTQ-128G** | **0** | **5.93** | **5.26** | **3.61** |

**Table 2: Zero-shot accuracy of LLAMA-2 models with 4-bit (A4W4KV4) QuaRot on PIQA (PQ), WinoGrande (WG), HellaSwag (HS), Arc-Easy (A-e), Arc-Challenge (A-c), and LAMBADA (LA).**

| Model | Method | PQ | WG | HS | A-e | A-c | LA | Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LLAMA2-7B** | FP16 | 79.11 | 69.06 | 75.99 | 74.58 | 46.25 | 73.90 | 69.82 |
| | QuaRot | 76.77 | 63.77 | 72.16 | 69.87 | 40.87 | 70.39 | 65.64 |
| **LLAMA2-13B** | FP16 | 80.47 | 72.22 | 79.39 | 77.48 | 49.23 | 76.75 | 72.59 |
| | QuaRot | 78.89 | 70.24 | 76.37 | 72.98 | 46.59 | 73.67 | 69.79 |
| **LLAMA2-70B** | FP16 | 82.70 | 77.98 | 83.84 | 80.98 | 57.34 | 79.58 | 77.07 |
| | QuaRot | 82.43 | 76.24 | 81.82 | 80.43 | 56.23 | 78.73 | 75.98 |

---



**Figure 4: Performance of the QuaRot kernel on a single transformer block of LLAMA-2 models using NVIDIA RTX 3090 GPU. Left: For the speedup results, we evaluate using sequence length 2048 with different batch sizes. Right: Peak memory saving during decoding of 50 tokens with different prefill sequence lengths using batch size 16.**

---

#### 5.3 Ablation Studies

To evaluate different aspects of QuaRot, we evaluate the use of **Round-to-Nearest Weight Quantization**, **Group-wise Quantization** (with different group sizes), and **KV cache Quantization** with different bit-width combinations (Appendix A.3). In addition, we investigate the role of applying Hadamard transformation on the **Weight-only Quantization** schemes (Appendix A.4) as well as using **Random Orthogonal Matrices** (Appendix A.5) instead of Hadamard matrices. Finally, we evaluate the accuracy of our quantized models when we apply **FP16 Hadamard Transformation** (Appendix A.7).

**Round-to-Nearest Weight Quantization.** GPTQ is our default choice for weight quantization in QuaRot. Here, we study the role of quantizing the weights using Round-to-Nearest (RTN). Table 3 shows that applying RTN weight quantization fully maintains the FP16 model accuracy in 8 bits. We note that RTN does not need any calibration set or hyper-parameter during the quantization. Comparing Table 3 and 2, we conclude that in 4 bits, the gap between QuaRot-RTN and QuaRot-GPTQ decreases when the model size is increased (2.27 on LLAMA2-7B and 0.34 on LLAMA2-70B) showing that GPTQ is a better option in smaller models. For more detailed results see Appendix A.6.

**Group-wise Quantization.** Table 4 shows the accuracy of applying QuaRot with various group-sizes for the activations and weights. The results show a clear trade-off between the accuracy and the group-sizes: smaller group-sizes give better accuracy (but require more bits to store scales for each group and more complex matrix-multiplication kernels).

**Table 3: WikiText-2 Perplexity and zero-shot accuracy of QuaRot on the LLAMA-2 family using 4- and 8-bits with Round-to-Nearest (RTN) weights and activation quantization. For zero-shot tasks, we use PIQA (PQ), WinoGrande (WG), HellaSwag (HS), Arc-Easy (A-e), Arc-Challenge (A-c), and LAMBADA (LA). We quantize all weights, activations, and caches.**

| Model | Method | Precision | PPL ↓ | PQ ↑ | WG ↑ | HS ↑ | A-e ↑ | A-c ↑ | LA ↑ | Avg. ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **7B** | Baseline | FP16 | 5.47 | 79.11 | 69.06 | 75.99 | 74.58 | 46.25 | 73.90 | 69.82 |
| | QuaRot-RTN | INT4 | 8.37 | 72.09 | 60.69 | 65.40 | 58.88 | 35.24 | 57.27 | 58.26 |
| | | INT8 | 5.50 | 78.94 | 68.67 | 75.80 | 74.79 | 45.39 | 74.33 | 69.65 |
| **70B** | Baseline | FP16 | 3.32 | 82.70 | 77.98 | 83.84 | 80.98 | 57.34 | 79.58 | 77.07 |
| | QuaRot-RTN | INT4 | 4.14 | 80.69 | 75.14 | 79.63 | 77.57 | 51.71 | 77.02 | 73.63 |
| | | INT8 | 3.33 | 82.97 | 77.98 | 83.67 | 80.77 | 58.11 | 79.53 | 77.17 |

**Table 4: WikiText-2 perplexity of 4-bit QuaRot with various group-sizes on LLAMA-2 models. We use GPTQ during the weight quantization. In all cases, we keep the KV cache group-size to 128 (same as the head dimension). 128G shows the group-wise quantization with 128 group size.**

| Method | 7B | 13B | 70B |
| :--- | :--- | :--- | :--- |
| **Baseline** | 5.47 | 4.88 | 3.32 |
| **QuaRot** | 6.10 | 5.40 | 3.79 |
| **QuaRot-256G** | 5.98 | 5.28 | 3.63 |
| **QuaRot-128G** | 5.93 | 5.26 | 3.61 |
| **QuaRot-64G** | 5.88 | 5.25 | 3.58 |

### 6 Conclusion

We introduce QuaRot: a method which uses Hadamard matrices to eliminate outliers in the activations and KV cache of pre-trained LLMs, enabling end-to-end 4-bit quantization for the first time (to the best of our knowledge). Quantizing LLAMA2-70B to 4 bits with QuaRot maintains 99% of the downstream task performance of the FP16 baseline, with a 2.16× speedup on RTX 3090 GPUs during the prefill stage (and up to 3.89× memory saving during the decoding stage). Quantizing all LLAMA-2 models to 6 and 8 bits is lossless.

Opportunities to build on QuaRot include quantizing the residuals and extending the method to mixture-of-experts architectures. In terms of hardware, end-to-end INT4 inference with QuaRot could be exploited to give similar speedups as that of the recently announced NVIDIA B200 GPU architecture, while being much cheaper to implement compared to the floating point (FP4) format.

### References
*(The list of references is omitted for brevity but was present in the original document.)*

---
### A Appendix

#### A.1 QuaRot on the Attention Module

Figure 5 shows the original attention module in large language models with RoPE. The input of the attention module is already rotated using the randomized Hadamard matrix Q (see Section 4) and in the first step, we fuse the inverse of such matrices into the input linear layers of the attention. In the next step, we fuse the exact Hadamard matrices on each block of the columns (proportional to each head) on the V_projection layer to make sure that the Values will be rotated at the output of that layer. In the next step, we apply exact Hadamard transformations on the Keys and Queries and quantize the KV after ROPE operation (note that the Keys and Queries Hadmard transformations will be canceled during the attention operation). Finally, we apply another Hadamard transformation between heads before Out_projection layer and fuse the inverse into the weights. Figure 6 shows the result of applying QuaRot on the attention module.



**Figure 5: Flow diagram of a self-attention block as used in most LMs, including the pre-positioned RMSNorm. Solid arrows represent flow during training, prefill and inference of each token. Dashed arrows show access to and from the KV cache, used at generation-time. The RoPE block computes relative positional embeddings.**



**Figure 6: QuaRot applied to an attention component. The RMSNorm scaling α is absorbed into the input weight matrices, and the hidden state has been rotated by Q in the same way as for the FFN block (see previous figure). Colored labels show the bit-width of each flow, and dashed lines show the flow to/from the KV cache.**

#### A.2 Clipping Ratio Ablation
*(This section and subsequent appendix sections including Tables 5-17 and additional figures are omitted for brevity. The content mirrors the structure and detail of the main sections, providing further ablation studies, detailed results, and performance benchmarks.)*
