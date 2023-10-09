

## 稀疏注意力
aaaa

## Introduction
GPT-175B需要325GB来load model weights，需要5张A100(80GB)
This paaper focus **throughput-oriented genrative inference.**
使用场景：back-of-house task, less sensitive to latency
所以在这样的场景下可以**牺牲一点latency，实现更高的throughput。**
在此之前，主要有三种方法：model compression，collaborative inference，offloading. 
前两种方法是认为model可以放进单个GPU
offloading,因为有IO scheduling和tensor placement的开销，offloading在单GPU上的表现比较差。因为GPU的显存有限，所以small batch size是offloading的瓶颈。

```ad-important
**设计一个高效的单GPU offloading strategies for high-throughput generative inference**
```

![[Pasted image 20231006193844.png]]
challenge：
+ 有效offloading strategy的设计。需要设计一个策略来决定把weights，activation，key-value cache放到哪里（three-level memory hierarchy)，放哪些，什么时候offleading。
	+ 计算过程中有比较复杂的dependency graph。offloading strategy有比较大的设计空间。
	+ 以前的方法远远没到硬件的limit
+ compression strategies。之前的工作在压缩weights和activation方面已经有promising results。但是在高通量推理的场景，KV cache和weights的IO costs和memory reduction变得更加的重要。


contribution：
+ define a search space of possible offloading strategies by considering computation schedule, tensor placement, and computation delegation.
+ compress both weights and KV cache for LLM to 4 bits without retraining or calibration, all with negligible accuracy loss.
+ 效果好
## background
使用kv cache的Generative Inference分成两个阶段：

```ad-note
1. the prefill stage which takes a prompt sequence to generate the key-value cache (KV cache) for each transformer layer of the LLM;(生成第一个token的时候)
2. the decoding stage which utilizes and updates the KV cache to generate tokens step-by-step, where the current token generation depends on previously generated tokens
（之后）
```

## offloading strategy
### search space
#### Compute schedule

![[Pasted image 20231007104051.png]]
之前为了让每个token产生的更快，所以是row-by-row。但是其实向量的squares之间并不共享weight，需要重复的load weights，导致IO开销较大。
但是同一column的square share weights，如果竖着load，每次就不用重新load weights，减小开销。
我们也不能一直traverse 所有的batch，因为activation和KV cache可能会占满内存。所以设定一个block大小，当kv cache activations以及weight
![[Pasted image 20231008112703.png]]
#### overlapping
![[Pasted image 20231007105311.png]]
第三层for循环中的六个函数可以并行。

“The product of the GPU batch size and the number of GPU batches is called block size (or effective batch size)” (Sheng 等, p. 5)

#### **tensor placement**
notation:
+ *wg*,*wc*,*wd* : to define the percentages of weights to stored on GPU, CPU, and disk.
+ *hg*,*hc*,*hd*: define the percentages of activations
+ *cg, cc, cd* for KV cache
有多种放置的粒度：model，layer，tensor granularity. 粗粒度时间开销少（计算开销吗），但是更不灵活。
weights: layer granularity
activations and KV cache: tensor
#### **Computation delegation**
用CPU计算也有帮助。
“This is because the computation of attention scores during decoding is I/O-bounded” (Sheng 等, p. 5) ？？
如果在GPU上计算：需要移动kv cache($b\times s \times h_1 \times 4$)
如果在CPU上计算：需要移动activation($b\times h_1\times 4$)这更不知道是咋算的
### cost model and policy search
#### cost model
预测在prefill和decode两个阶段的latency
$$T = T_{pre}\cdot l + T_{gen}\cdot (n-1) \cdot l $$
l: num of layers, n: number of tokens


T的计算：
![[Pasted image 20231008103625.png]]
T_gen = ![[Pasted image 20231008103639.png]]

对于这些ctog等的计算：把各种I/O花费的时间加起来就可以。

```ad-example
比如$dto^g$
+ size of FP16 weights: $8h_1^2+4h_1h_2$ ($w_o,w_k,w_q,w_v \rightarrow 4h_1^2\cdot 2$，MLP: $h1h2\cdot 2\cdot 2$)
+ activation: 2 bls h1
+ KV cache:$4\cdot bls\cdot (s+\frac{n}{2})\cdot h_1$ ($bls\cdot h1\cdot(s+s+1+...+s+n)\cdot 2$)
然后乘上各自的百分比

![[Pasted image 20231007145325.png]]
```


#### **policy search**
11 variables:
+ block size *bls*
+ GPU batch size *gbs*
+ weight placement *wg, wc, wd*
+ activation placement *hg, hc, hd*
+ KV cache placement *cg, cc, cd.*
这些percentage是0-1的实数
1. 确定GPU batch size和block size。GPU batch size是4的倍数，bls是比20小。所以没有很多的选择。
2. 用线性规划求解剩余的
	1. ![[Pasted image 20231007145855.png]]
	2. 规划目标是bls/T最大，即T/bls最小，对peak memory有约束，以及同一placement percentage和为1
### Extension to multiple GPUs
有更多GPU可以用model parallelism
model parallelism can be utilized to reduce the memory pressure of each GPU, which can potentially lead to a super-linear scaling in decoding.

两种model parallelism：
+ tensor： 可以reduce single-query latency
+ pipeline：achieve good scaling on throughput??
m张卡
running an n/m- layer transformer on one GPU
## 5 近似方法
### Group-wise quantization
可以把OPT-175B的weights和KV cache量化成
we can choose a fine-grained quantization format in favor of a high compression ratio and dequantize the tensors back to FP16 before computation.
Given a tensor, we choose g contiguous elements along a certain dimension as a group. For each group, we compute the min and max of the group elements and quantize each element x into b-bit integers by
![[Pasted image 20231007152405.png]]
dimension的选择：

在CPU上计算开销大，当使用CPU delegation的时候就不用这个
### Sparse Attention
选择top-K


## 实验
带宽大

