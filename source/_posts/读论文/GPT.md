参考：
https://jalammar.github.io/illustrated-gpt2/
[60行代码构建gpt](https://jaykmody.com/blog/gpt-from-scratch/)
https://jiqihumanr.github.io/2023/04/13/gpt-from-scratch/
[机器学习术语表](https://developers.google.com/machine-learning/glossary?hl=zh-cn#logits )
https://jalammar.github.io/illustrated-gpt2/

咱只看decoder
定义一下参数：
+ hidden size: 每个词的embedding的维度
+ seq len
和一些常用的英文表示：
+ wte: 词向量嵌入矩阵， shape为`[n_vocab, n_emdb] `
+ wpe: 位置嵌入，`[n_ctx, n_emdb] `
input是`[n_seq, n_embd]`
```
x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
```
logits:可以理解logits ——【batch_size，class__num】是未进入softmax的概率，一般是全连接层的输出，softmax的输入
## 到底有哪些层
![[Pasted image 20230927144307.png]]
### embedding
1. 就是上述的wte和wpe
### transformer layer
![[Pasted image 20230927141646.png]]
```python
""" 代码转载自https://jiqihumanr.github.io/2023/04/13/gpt-from-scratch/
"""
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]  
    # multi-head causal self attention  
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]  
  
    # position-wise feed forward network  
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]  
  
    return x

```
注意点：
1. multi-head attention: 建模输入之间的关系。除此之外，别的都是对单一token进行的，不会看到彼此。
2. ffn只是为了给模型增加可学习的参数
3. 预归一化：x + sublayer(layer_norm(x))

对于transformer layer:

什么是自注意力：qkv来自同一个来源，就是自注意力
注意这个mask
```python
def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]  
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v  
  
def causal_self_attention(x, c_attn, c_proj): # [n_seq, n_embd] -> [n_seq, n_embd]  
    # qkv projections  
    x = linear(x, **c_attn) # [n_seq, n_embd] -> [n_seq, 3*n_embd]  
  
    # split into qkv  
    q, k, v = np.split(x, 3, axis=-1) # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]  
  
    # causal mask to hide future inputs from being attended to  
    causal_mask = (1 - np.tri(x.shape[0]), dtype=x.dtype) * -1e10  # [n_seq, n_seq]  
  
    # perform causal self attention  
    x = attention(q, k, v, causal_mask) # [n_seq, n_embd] -> [n_seq, n_embd]  
  
    # out projection  
    x = linear(x, **c_proj) # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]  
  
    return x
```

层归一化
