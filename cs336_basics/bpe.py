# bpe.py
# byte-pair tokenizer

from collections.abc import Iterable
import os
import json
from typing import Iterator, Optional
import regex as re

type Pretoken = tuple[bytes, ...]
type BytePair = tuple[bytes,bytes]
type BytePairCounts = dict[BytePair, int]
type ByteCounts = dict[BytePair, int]

# return the byte-pair counts from a given byte sequence
def pretoken_byte_pair_counts(pretoken: tuple[bytes, ...]) -> BytePairCounts:
    counts: dict[tuple[bytes,bytes], int] = {}
    for i in range(len(pretoken)-1):
        bytepair = (pretoken[i], pretoken[i+1])
        counts.setdefault(bytepair, 0)
        counts[bytepair] += 1

    return counts

def merge_byte_pair(pretoken: Pretoken, byte_pair: BytePair) -> Pretoken:
    new_pretoken_bytes: list[bytes] = []
    counts: BytePairCounts = dict()

    n = len(pretoken)
    i = 0
    while i < n:
        if i < n - 1 and (pretoken[i], pretoken[i+1]) == byte_pair:
            merged_byte_pair = b''.join(byte_pair)
            new_pretoken_bytes.append(merged_byte_pair)
            i += 2

        else:
            new_pretoken_bytes.append(pretoken[i])
            i += 1

    new_pretoken = tuple(new_pretoken_bytes)
    return new_pretoken

def to_pretoken(bs: bytes) -> Pretoken:
    return tuple(b.to_bytes(1) for b in bs)

def pretokenize(file_text: bytes, special_tokens: list[str]) -> dict[Pretoken, int]:
    special_tokens_pattern: re.Pattern[bytes] = re.compile("|".join(map(re.escape, special_tokens)).encode("utf-8"))
    text_list: list[bytes] = re.split(special_tokens_pattern, file_text)

    # next, pre-tokenize text
    # this is the regex used in GPT-2
    pretoken_counts: dict[Pretoken, int] = {}
    pretoken_pat: re.Pattern[bytes] = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""".encode("utf-8"))
    for text in text_list:
        for pretoken_match in re.finditer(pretoken_pat, text):
            pretoken: Pretoken = to_pretoken(pretoken_match[0])

            if pretoken not in pretoken_counts:
                pretoken_counts[pretoken] = 1

            else:
                pretoken_counts[pretoken] += 1

    return pretoken_counts

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    with open(input_path, "rb") as f:
        file_text: bytes = f.read()

    vocab: dict[int, bytes] = dict((i, i.to_bytes(1)) for i in range(256))
    merges: list[tuple[bytes,bytes]] = []

    next_vocab_item: int = 256
    for special_token in special_tokens:
        vocab[next_vocab_item] = special_token.encode("utf-8")
        next_vocab_item += 1

    pretoken_counts: dict[Pretoken, int] = pretokenize(file_text, special_tokens)

    # map from byte-pairs to pretokens that contain the byte-pair
    byte_pair_index: dict[BytePair, set[Pretoken]] = dict()

    # map from original pretoken to current merged pretoken
    pretoken_versions: dict[tuple[bytes, ...], tuple[bytes, ...]] = {}

    # byte pair counts for current pretoken versions
    cur_pretoken_byte_pair_counts: dict[Pretoken, BytePairCounts] = dict()

    # count byte-pairs by counting them on each pre-token
    byte_pair_counts: BytePairCounts = dict()

    # populate pretoken data structures
    for pretoken, pretoken_count in pretoken_counts.items():
        pretoken_versions[pretoken] = pretoken
        counts = pretoken_byte_pair_counts(pretoken)
        cur_pretoken_byte_pair_counts[pretoken] = counts
        for byte_pair, byte_pair_count in counts.items():
            byte_pair_counts.setdefault(byte_pair, 0)
            byte_pair_counts[byte_pair] += byte_pair_count * pretoken_count

            byte_pair_index.setdefault(byte_pair, set())
            byte_pair_index[byte_pair].add(pretoken)

    # merge frequent byte-pairs until we reach the desired vocab size
    while next_vocab_item < vocab_size:
        # compute byte pair to merge
        # byte-pairs are ordered as follows: 
        # - first, by frequency;
        # - if there are multiple maximal-count byte-pairs, pick the
        #   lexicographically-maximal byte pair
        #
        # we take advantage of the fact that tuples are naturally ordered
        # lexicographically to enforce this ordering
        max_item = max(map(lambda t: t[::-1], byte_pair_counts.items()))
        max_byte_pair: tuple[bytes, bytes] = max_item[1] # type: ignore
        new_token: bytes = b''.join(max_byte_pair)

        updated_pretokens = byte_pair_index[max_byte_pair].copy()
        for pretoken in updated_pretokens:
            # update the current version of the pretoken so that
            # current byte-pair is merged
            cur_pretoken = pretoken_versions[pretoken]
            new_pretoken = merge_byte_pair(cur_pretoken, max_byte_pair)

            pretoken_count = pretoken_counts[pretoken]
            pretoken_versions[pretoken] = new_pretoken

            # remove the current version of the pretoken from byte-pair counts 
            # and byte-pair index
            cur_counts = cur_pretoken_byte_pair_counts[pretoken]
            for byte_pair, byte_pair_count in cur_counts.items():
                byte_pair_counts[byte_pair] -= byte_pair_count * pretoken_count
                byte_pair_index[byte_pair].remove(pretoken)

            # add new version of pretoken to byte-pair counts and byte-pair index
            new_counts = pretoken_byte_pair_counts(new_pretoken)
            for byte_pair, byte_pair_count in new_counts.items():
                if byte_pair in byte_pair_counts:
                    byte_pair_counts[byte_pair] += byte_pair_count * pretoken_count
                    byte_pair_index[byte_pair].add(pretoken)

                else:
                    byte_pair_counts[byte_pair] = byte_pair_count * pretoken_count
                    byte_pair_index[byte_pair] = { pretoken }

            cur_pretoken_byte_pair_counts[pretoken] = new_counts

        # add new token to vocab
        merges.append(max_byte_pair)
        vocab[next_vocab_item] = new_token
        next_vocab_item += 1

    return vocab, merges

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[BytePair], special_tokens: list[str] | None=None):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[BytePair] = merges
        self.special_tokens: list[str] = special_tokens if special_tokens is not None else []
        self.vocab_index = { v: i for i, v in self.vocab.items() }

        self.secondary_special_tokens = \
            [t1 for t1 in self.special_tokens for t2 in self.special_tokens if t1 != t2 and t1 in t2]

        if len(self.secondary_special_tokens) > 0:
            secondary_special_tokens_pattern_str = "(" + "|".join(map(re.escape, self.secondary_special_tokens)) + ")"
            self.secondary_special_tokens_pattern: re.Pattern[bytes] = \
                re.compile(secondary_special_tokens_pattern_str.encode("utf-8"))

        self.primary_special_tokens = \
            [t for t in self.special_tokens if t not in self.secondary_special_tokens]

        if len(self.primary_special_tokens) > 0:
            primary_special_tokens_pattern_str = "(" + "|".join(map(re.escape, self.primary_special_tokens)) + ")"
            self.primary_special_tokens_pattern: re.Pattern[bytes] = \
                re.compile(primary_special_tokens_pattern_str.encode("utf-8"))

    def from_files(_cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab: dict[int, bytes] = dict()
        with open(vocab_filepath, "r") as f:
            vocab_index: dict[bytes, int] = json.load(f)
            for token, index in vocab_index.items():
                vocab[index] = token

        merges: list[tuple[bytes,bytes]] = []
        with open(merges_filepath) as f:
            for lines in f:
                line_bytes: bytes = lines.encode("utf-8")
                first_token: bytes = b''
                for b in line_bytes:
                    if first_token in vocab_index:
                        break

                    first_token += b.to_bytes(1)

                second_token = line_bytes[len(first_token):]
                merges.append((first_token, second_token))
                        
        return Tokenizer(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        if len(self.primary_special_tokens) > 0:
            text_list: list[bytes] = re.split(self.primary_special_tokens_pattern, text.encode("utf-8"))

        else:
            text_list: list[bytes] = [text.encode("utf-8")]

        # next, pre-tokenize text
        # this is the regex used in GPT-2
        tokens = []
        pretoken_pat: re.Pattern[bytes] = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""".encode("utf-8"))
        for pretoken_bytes in text_list:
            # if pretoken_bytes is in the vocab index, it's a special token
            if pretoken_bytes.decode("utf-8", errors="replace") in self.primary_special_tokens:
                tokens.append(self.vocab_index[pretoken_bytes])
                continue

            if len(self.secondary_special_tokens) > 0:
                secondary_list = re.split(self.secondary_special_tokens_pattern, pretoken_bytes)           

            else:
                secondary_list = [pretoken_bytes]

            for secondary_pretoken in secondary_list:
                if secondary_pretoken.decode("utf-8", errors="replace") in self.secondary_special_tokens:
                    tokens.append(self.vocab_index[secondary_pretoken])
                    continue

                for pretoken_match in re.finditer(pretoken_pat, secondary_pretoken):
                    pretoken: Pretoken = to_pretoken(pretoken_match[0])

                    for byte_pair in self.merges:
                        pretoken = merge_byte_pair(pretoken, byte_pair)

                    tokens.extend(self.vocab_index[bs] for bs in pretoken)

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token in self.encode(text):
                yield token

    def decode(self, ids: list[int]) -> str:
        bs: bytes = b''.join(self.vocab[id] for id in ids)
        return bs.decode("utf-8", errors="replace")
