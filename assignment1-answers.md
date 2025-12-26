# Assignment 1 - CS 336

## 2.1 - The Unicode Standard

**Problem (unicode1): Understanding Unicode**

a. What Unicode character does `chr(0)` return?

- `\x00`

b. How does this character's string representation (`__repr__()`) differ from
   its printed representation?

- It doesn't get printed as it's a non-printable character

c. What happens when this character occurs in text?

- It gets shown its string representation "\x00"

## 2.2 - Unicode Encodings

*Why don't we just use UTF-8 or some other Unicode encoding to tokenize?"*

UTF-8 and others map characters to codepoints (integers), exactly what
tokenization is supposed to do. The problem is that using UTF-8 or something
like it directly would create a very sparse vocabulary; There are over 150k
codepoints in Unicode, and most of those characters are rarely used. (There's
codepoints for lots of random symbols!)

*Why don't we just use the byte representation of UTF-8 to tokenize?*

That would mean a vocab size of 256, which would blow up sequence lengths
and make training and inference very inefficient.

**Problem (unicode2): Unicode Encodings**

a. What are some reasons to prefer training our tokenizer on UTF-8 encoded
   bytes, rather than UTF-16 or UTF-32?

- UTF-16 and UTF-32 sacrifice encoding length to make the encodings more
  fixed-length (UTF-32 is totally fixed-length, UTF-16 can either have 1 or 2
  16-bit units). These encodings have more "empty-space" than UTF-8: they
  take up more bytes to encode the same character sequence, and thus more
  tokens to represent the same character sequence.

b. Why is this function incorrect? Provide an example of an input byte string
   that yields incorrect results.

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")) 'hello'
```

- UTF-8 is variable-length, meaning some characters take up more than one byte.
  Decoding individual bytes of such characters would generate an exception,
  because they can't be decoded. For example:

```python
>>> decode_utf8_bytes_to_str_wrong("好".encode("utf-8"))

UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe5 in position 0: unexpected end of data
```

c. Give a two byte sequence that does not decode to any Unicode characters (s).

- `0x88 0x88`. Why? Byte 1 of any Unicode character starts with either `0`,
  `110`, `1110`, or `11110`. `0x88` is `0b10001000`.

## 2.3 - Subword Tokenization

Subword tokenization is the mid-point b/w byte-level and word-level encoding;
it trades off the small vocab size of byte-level encoding for better
compression.

Byte-pair encoding: a compression algorithm that iteratively replaces frequent
byte-pairs with a new index in the vocabulary.

## 2.4 - Byte-Pair Tokenizer Training

Steps to BPE tokenizer training

1. **Vocab initialization**. Start with byte-level encoding. Initial vocab size
   would then be 256.

2. **Pre-tokenization**. Split up the input corpus in a coarse-grained manner
   using a pre-tokenization scheme. In GPT-2, a regex was used to split English
   text into words, with sensitivity towards contractions and punctuation.

3. **Compute BPE merges**. Do the following:

   - Compute byte-pair frequences for each pre-token.
   
   - Sum pre-token frequencies where byte-pair occurs to compute total byte-pair
   frequency, then merge. If multiple byte-pairs have maximal frequencies, pick
   the one to merge by lexicographic order.
   
   - Repeat until you reach desired vocab size.

   - Should handle special tokens (e.g. `<|endoftext>`) and preserve them as
     single tokens.

# 3.6 - The Full Transformer LM

## Counting Required FLOPS for Transformer Model's Forward Pass

Matrix multiples dominate the computational requirements of a transformer model,
so we only need to count matmuls to estimate how many FLOPS are required to do
a forward pass.

Rule of thumb: matrix multiply between AxB and BxC matrices takes 2ABC FLOPS.

Computation per layer:

**Transformer Block (Multihead Self-Attention + SwiGLU FFN)**

(Remember: d_embed = d_model / heads)

1. Q,K,V Projection from d_model to d_embed:

- matrix 1: seq x d_model
- matrix 2 (WQKV): d_model x (3 * heads * d_embed)
- FLOPs: batch * 2 * seq * d_model 3 * heads * d_embed
       = batch * 6 * seq * d_model * heads * d_embed
       = batch * 6 * seq * d_model^2

2. RoPE for Q and K vectors:

- matrix 1: (2 * batch * seq * heads * d_embed/2) x 2
- matrix 2: 2 x 2
- output: (batch * seq * heads * d_embed/2) x 2
- FLOPS: 4 * batch * seq * heads * d_embed/2 * 2 * 2
       = 8 * batch * seq * d_model

3. Scaled Dot-Product Attention with Causal Mask:

3.1. Scaled Attention Score (softmax(QK^T / sqrt(d_embed)))

- matrix 1 (Q): seq x d_embed
- matrix 2 (K): d_embed x seq
- output: seq x seq
- FLOPS: batch * heads * 2 * seq^2 * d_embed
       = 2 * batch * seq^2 * d_model

3.2. Compute Attention-Value Product:

- matrix 1: seq x seq
- matrix 2: seq x d_embed
- output: seq * d_embed
- FLOPS: batch * heads * 2 * seq * seq * d_embed
       = 2 * batch * seq^2 * d_model

3.3. Total for Multihead Self-Attention:

2 * 2 * batch * seq^2 * d_model
= 4 * batch * seq^2 * d_model

4. Output Projection to d_model:

- matrix 1: seq x (heads * d_embed)
- matrix 2 (WO): (heads * d_embed) x d_model
- FLOPs: batch * 2 * seq * heads * d_embed * d_model
       = 2 * batch * seq * d_model^2

5. SwiGLU FFN

SwiGLU FFN is W2 * (HadamardProduct(SiLU(W1 * x), W3 * x)

5.1. W1 * x
- matrix 1 (x): (batch * seq) x d_model
- matrix 2 (W1): d_model x d_ff
- output: (batch * seq) x d_ff
- FLOPS: 2 * batch * seq * d_model * d_ff

5.2. W3 * x

Same accounting as W1 * x.

5.3. W2 * (HadamardProduct(SiLU(W1 * x), W3 * x)

- matrix 1: (batch * seq) x d_ff
- matrix 2 (W2): d_ff x d_model
- output: (batch * seq) x d_model
- FLOPS: 2 * batch * seq * d_ff * d_model

5.4 Total for SwiGLU FFN:
3 * (2 * batch * seq * d_ff * d_model)
= 6 * batch * seq * d_ff * d_model

Assuming d_ff = 8/3 * d_model:
6 * batch * seq * d_ff * d_model
= 16/3 * batch * seq * d_model^2

Total for Transformer Block:

- QKV projection from d_model to d_embed: batch * 6 * seq * d_model^2
- Q,K RoPE: 8 * batch * seq * d_model
- Multihead Self-Attention: 4 * batch * seq^2 * d_model
- Output Projection to d_model: 2 * batch * seq * d_model^2
- SwiGLU FFN: 6 * batch * seq * d_ff * d_model

2 * batch * seq * d_model * ((3 * d_model) + 4 + (2 * seq) + d_model + (3 * d_ff))
= 2 * batch * seq * d_model * ((4 * d_model) + (2 * seq) + (3 * d_ff) + 4)

**Output Projection to Vocab**

- matrix 1: (batch * seq) x d_model
- matrix 2: d_model x vocab_size
- output: (batch * seq) x vocab_size
- FLOPS: 2 * batch * seq * d_model * vocab_size

**Total FLOPS**

layers * 2 * batch * seq * d_model * ((4 * d_model) + (2 * seq) + (3 * d_ff) + 4)
+ 2 * batch * seq * d_model * vocab_size
=
2 * batch * seq * d_model * (
   (layers * ((4 * d_model) + (2 * seq) + (3 * d_ff) + 4)) + vocab_size
)

## Number of Parameters for Transformer Model

Embedding:
- total params: vocab_size * d_model

Transformer Block:
- Multihead Self-Attention:
   - WQKV: d_model x (3 * heads * d_embed)
   - WO: (heads * d_embed) x d_model
   - total params: 3 * d_model^2 + d_model^2 = 4 * d_model^2

- SwiGLU FFN:
   - W1, W2, W3: d_model x d_ff
   - total params: 3 * d_model * d_ff

- total params: d_model * (4 * d_model + 3 * d_ff)

Output Projection:
- total params: d_model x vocab_size

Total params:
vocab_size * d_model
+ layers * (d_model * (4 * d_model + 3 * d_ff))
+ d_model x vocab_size

## Problems

**Problem (transformer_accounting): Transformer Accounting**

a. Consider GPT-2 XL, which has the following configuration:

- vocab_size : 50,257
- context_length : 1,024
- num_layers : 48
- d_model : 1,600
- num_heads : 25
- d_ff : 6,400

Suppose we constructed our model using this configuration. How many trainable
parameters would our model have? Assuming each parameter is represented using
single-precision floating point, how much memory is required to just load this
model?

```
number of params =
vocab_size * d_model
+ layers * (d_model * (4 * d_model + 3 * d_ff))
+ d_model x vocab_size
=
50257 * 1600
+ 48 * (1600 * (4 * 1600 + 3 * 6400))
+ 1600 * 50257
=
2,126,902,400 params

If each param is stored as 32-bit single precision:
2126902400 * 4 bytes =
8,507,609,600 bytes =
8.508 GB
```

b. How many FLOPs do these matrix multiplies require in total?

```
2 * batch * seq * d_model * (
   (layers * ((4 * d_model) + (2 * seq) + (3 * d_ff) + 4)) + vocab_size
)
=
2 * 1024 * 1600 * (
   (48 * ((4 * 1600) + (2 * 1024) + (3 * 6400) + 4)) + 50257
)
=
4,513,965,670,400 FLOPS
~4.5 TFLOPS
```

c. Based on your analysis above, which parts of the model require the most FLOPs?

- The SwiGLU FFN components of transformer layers. See:

SwiGLU FFN FLOPS:
```
layers * 6 * batch * seq * d_ff * d_model
=
48 * 6 * 1024 * 6400 * 1600
=
3,019,898,880,000
~3,020 GFLOPS
```

Attention FLOPS (no RoPE, embedding/output projection)
```
layers * 4 * batch * seq^2 * d_model
=
48 * 4 * 1024^2 * 1600
=
322,122,547,200
~322 GFLOPS
```

d. Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads),
GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers,
1280 d_model, 20 heads). As the model size increases, which parts of the
Transformer LM take up proportionally more or less of the total FLOPs?

GPT-2 small (12 layers, 768 d_model, 12 heads)
```
2 * batch * seq * d_model * (
   (layers * ((4 * d_model) + (2 * seq) + (3 * d_ff) + 4)) + vocab_size
)
=
2 * 1 * 1024 * 768 * (
   (12 * ((4 * 768) + (2 * 1024) + (3 * 6400) + 4)) + 50257
)
=
538,147,553,280 FLOPs
```

GPT-2 medium (24 layers, 1024 d_model, 16 heads)
```
2 * batch * seq * d_model * (
   (layers * ((4 * d_model) + (2 * seq) + (3 * d_ff) + 4)) + vocab_size
)
=
2 * 1 * 1024 * 1024 * (
   (24 * ((4 * 1024) + (2 * 1024) + (3 * 6400) + 4)) + 50257
)
=
1,381,203,181,568 FLOPs
```

GPT-2 large (36 layers, 1280 d_model, 20 heads)
```
2 * batch * seq * d_model * (
   (layers * ((4 * d_model) + (2 * seq) + (3 * d_ff) + 4)) + vocab_size
)
=
2 * 1 * 1024 * 1280 * (
   (36 * ((4 * 1280) + (2 * 1024) + (3 * 6400) + 4)) + 50257
)
=
2,620,519,874,560 FLOPs
```

e. Take GPT-2 XL and increase the context length to 16,384. How does the total
FLOPs for one forward pass change? How do the relative contribution of FLOPs of
the model components change?

```
2 * batch * seq * d_model * (
   (layers * ((4 * d_model) + (2 * seq) + (3 * d_ff) + 4)) + vocab_size
)
=
2 * 16384 * 1600 * (
   (48 * ((4 * 1600) + (2 * 16384) + (3 * 6400) + 4)) + 50257
)
=
149,532,862,054,400 FLOPS
~149.5 TFLOPS
```

The FLOPS required from 1,024 context size to 16,384 (16x increase)
increased ~35x!

SwiGLU FFN FLOPS:
```
layers * 6 * batch * seq * d_ff * d_model
=
48 * 6 * 1 * 16384 * 6400 * 1600
=
48,318,382,080,000
~48,318 GFLOPS
```

Attention FLOPS (no RoPE, embedding/output projection)
```
layers * 4 * batch * seq^2 * d_model
=
48 * 4 * 1 * 16384^2 * 1600
=
82,463,372,083,200
~82,463 GFLOPS
```

Attention now dominates the required FLOPs; required attention FLOPS increased
by ~256x while required SwiGLU FFN FLOPs increased by ~16x. Thus as context
size increases, attention becomes the computational bottleneck.

# 4.3. AdamW

**Problem (adamwAccounting): AdamW Accounting**

a. How much peak memory does running AdamW require? Decompose your answer based
on the memory usage of the parameters, activations, gradients, and optimizer
state. Express your answer in terms of the batch_size and the model
hyperparameters (vocab_size, context_length, num_layers,d_model,num_heads).
Assume `d_ff=4 x d_model`.

Parameters:
```
vocab_size * d_model
+ layers * (d_model * (4 * d_model + 3 * d_ff))
+ d_model x vocab_size
```

Gradients: same as parameters!

Optimizer state:
2 * parameters (first moment + second moment for each parameter)

Activations:
Transformer block
- RMSNorm(s): 2 * batch * seq * d_model
   - 1 for before attention, 1 for before FFN

– Multi-head self-attention
   - QKV projections: 3 * batch * seq * heads * d_embed = 3 * batch * seq * d_model
   - QK matrix multiply: batch * heads * seq * seq
   - softmax: batch * heads * seq * seq
   - weighted sum of values with V: batch * heads * seq * d_embed = batch * seq * d_model
   - output projection: batch * seq * d_model
   - total: 5 * batch * seq * d_model + 2 * batch * heads * seq * seq

– Position-wise feed-forward:
   - W1 matrix multiply: 4 * batch * seq * d_model
   - W3 matrix multiply: 4 * batch * seq * d_model
   - SiLU: 4 * batch * seq * d_model
   - W2 matrix multiply: batch * seq * d_model
   - total: 13 * batch * seq * d_model

- Total for transformer block:
    2 * batch * seq * d_model
  + 5 * batch * seq * d_model + 2 * batch * heads * seq * seq
  + 13 * batch * seq * d_model

Final RMSNorm: batch * seq * d_model

Output embedding: batch * seq * vocab_size

Cross-entropy on logits: 1

Total Activations:

layers * (
    2 * batch * seq * d_model
  + 5 * batch * seq * d_model + 2 * batch * heads * seq * seq
  + 13 * batch * seq * d_model
)
+ batch * seq * d_model
+ batch * seq * vocab_size
   
Peak memory usage:

- Peak usage is at the end of forward pass, before backward pass,
when all of activations must be kept.

- Total bytes used:

```
format size * (parameters + gradients + optimizer states + activations)
=
4 * (4 * parameters + activations)
=
4 * (4 * (
   vocab_size * d_model
   + layers * (d_model * (4 * d_model + 3 * d_ff))
   + d_model x vocab_size
)
+ layers * (
    2 * batch * seq * d_model
  + 5 * batch * seq * d_model + 2 * batch * heads * seq * seq
  + 13 * batch * seq * d_model
)
+ batch * seq * d_model
+ batch * seq * vocab_size)
```

b. Instantiate your answer for a GPT-2 XL-shaped model to get an expression that
only depends on the batch_size. What is the maximum batch size you can use and
still fit within 80GB memory?

Assuming single-precision for all values (FP32):

```
4 * (4 * (
   vocab_size * d_model
   + layers * (d_model * (4 * d_model + 3 * d_ff))
   + d_model x vocab_size
)
+ layers * (
    2 * batch * seq * d_model
  + 5 * batch * seq * d_model + 2 * batch * heads * seq * seq
  + 13 * batch * seq * d_model
)
+ batch * seq * d_model
+ batch * seq * vocab_size)
=
4 * (4 * (
   50257 * 1600
   + 48 * (1600 * (4 * 1600 + 3 * 6400))
   + 1600 x 50257
)
+ 48 * (
    2 * batch * 1024 * 1600
  + 5 * batch * 1024 * 1600 + 2 * batch * 25 * 1024 * 1024
  + 13 * batch * 1024 * 1600
)
+ batch * 1024 * 1600
+ batch * 1024 * 50257)
=
4 * (8507609600 + 4142547968 * batch)
=
4 * (8507609600 + 4142547968 * batch)
=
16570191872 * batch + 34030438400
```

Batch size that can fit in 80GB?
```
16570191872 * batch + 34030438400 = 80000000000
=
16570191872 * batch = 45969561600
~
2.77
```

So a batch size of a most 2 can fit in 80GB.