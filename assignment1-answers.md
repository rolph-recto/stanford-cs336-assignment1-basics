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
