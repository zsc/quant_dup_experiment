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

***

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

