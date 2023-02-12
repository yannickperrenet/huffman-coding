"""Huffman encoding and decoding.

`encode()` encodes a given stream fully such that `decode()` can decode
it (without needing the original text, because the frequency table is
stored in the encoding as well). The encoding format is as follows:

    - 1 byte: the number of (ASCII) characters in the frequency table.
    - X bytes: where X=5*value_of_previous_byte. These bytes store the
      frequency table using the schema given by
      `FREQ_TABLE_ENCODING_SCHEMA`.
    - Y bytes: the remaining bytes containing the encoded text.

To make sure the encoding of the text is "byte-aligned" we make use of a
`PSEUDO_EOF`. Once the `PSEUDO_EOF` is read during decoding, we stop
decoding further. Decoding is done using a Finite State Machine (FSM)
with a configurable `DECODER_WORD_SIZE` that determines the number of
transitions per state (`num_transitions = 1 << DECODER_WORD_SIZE`). This
makes decoding considerably faster as we no longer have to decode
bit-by-bit but can instead (at once) decode `DECODER_WORD_SIZE` number
of bits.

Todo:
    - Working with larger than memory input text.
    - The `DECODER_WORD_SIZE` can not be larger than 8 currently,
      because we decode byte-by-byte. To allow for larger values of
      `DECODER_WORD_SIZE` we would have to dynamically change the number
      of bytes that are read for decoding.
    - If `PSEUDO_EOF` is actually in the given text (to be encoded),
      then an error is raised. Although unlikely that the `PSEUDO_EOF`
      is in the text (currently a static uncommon character is chosen),
      if it is, then we error out. Alternatively, a new `PSEUDO_EOF` can
      be chosen (dynamically) that is not in the text. Only if all 256
      ASCII characters are in the text, then we error out.

"""

import heapq
import io
import itertools
import struct
import sys
import typing as t
from collections import defaultdict, deque
from enum import Enum


# This is to ensure we stop decoding instead of having to pad the
# output.
PSEUDO_EOF = chr(4)  # End Of Transmission

# Number of bits to process at once using a Finite State Machine (FSM).
# Recommend value is 4 as it has a good trade-off between speed and
# memory consumption. In addition, for large values it takes a
# considerable amount of time to build up the FSM.
DECODER_WORD_SIZE = 4

# Decoder flags.
DECODER_FAIL = 1
DECODER_COMPLETE = (1 << 1)

# Encoding schema using the `struct` module.
# To use, do: `eval(FREQ_TABLE_ENCODING_SCHEMA.format(num_chars=...))`
FREQ_TABLE_ENCODING_SCHEMA = "'<' + {num_chars} * 'cI'"


class Direction(Enum):
    """Code values for directions (left or right) in Huffman tree."""
    LEFT = 0
    RIGHT = 1


class TreeNode:
    def __init__(
        self,
        char: str = "",
        freq: int = 0,
        left: t.Optional["TreeNode"] = None,
        right: t.Optional["TreeNode"] = None,
    ) -> None:
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
        self.fsm_state: t.Optional[int] = None

    def __lt__(self, other):
        return self.freq < other.freq

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        # Wrap self.char in repr() as it could be "\n" or similar.
        return (
            "TreeNode("
                f"char={repr(self.char)}, freq={self.freq},"
                f" left={self.left}, right={self.right}"
            ")"
        )


def _get_freq_table(f_in: t.TextIO) -> dict[str, int]:
    freq_table = defaultdict(int)
    while (chars := f_in.read(io.DEFAULT_BUFFER_SIZE)):
        for char in chars:
            freq_table[char] += 1

    freq_table[PSEUDO_EOF] += 1
    if freq_table[PSEUDO_EOF] > 1:
        raise RuntimeError("Pseudo-EOF is actually in given text.")

    return freq_table


def _get_huffman_tree(freq_table: dict[str, int]) -> TreeNode:
    """Constructs a Huffman tree."""
    heap = []
    for char in freq_table:
        node = TreeNode(char=char, freq=freq_table[char])
        heap.append(node)

    heapq.heapify(heap)
    while len(heap) != 1:
        least_freq1, least_freq2 = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(
            heap,
            TreeNode(
                freq=least_freq1.freq + least_freq2.freq,
                left=least_freq1,
                right=least_freq2,
            ),
        )

    # Return root node.
    return heap[0]


def _get_huffman_code(root: TreeNode) -> dict[str, str]:
    """Constructs Huffman code given a Huffman tree.

    Returns:
        Maps a character in the original text (which corresponds to a leaf
        node in the Huffman tree) to its corresponding code, where the code
        is represented as a string of zeros "0" and ones "1".

    """
    def helper(root: TreeNode) -> None:
        nonlocal ans, codeword

        # Reached a leaf node.
        if root.left is None and root.right is None:
            ans[root.char] = "".join(codeword)
            return

        if root.left is not None:
            codeword.append(str(Direction.LEFT.value))
            helper(root.left)
            codeword.pop()

        if root.right is not None:
            codeword.append(str(Direction.RIGHT.value))
            helper(root.right)
            codeword.pop()

    ans = {}
    codeword = []
    helper(root)
    return ans


def _get_fsm_decoder(
    root: TreeNode,
    word_size: int = 4,
) -> list[tuple[int, int, str]]:
    """Gets an FSM that can be used as a Huffman decoder.

    Each internal node of the Huffman tree (including the root node)
    will become a state in the FSM. For each state we consider all
    possible transitions based on the specified `word_size`. For
    example, when `word_size=4` the possible transitions are the bit
    sequences: `00`, `01`, `10` and `11`.

    Todo:
        A possible performance improvement (at the cost of memory) would
        be to generate the FSM for increasing `word_size`s (as power of
        2) until the required `word_size` is reached, e.g. generate FSM
        for `word_size=2` and use that FSM to generate for
        `word_size=4`, etc.

    Returns:
        A single list containing all transitions for all states defining
        the FSM.

        Each state will be a sequence of (1 << word_size) number of
        tuples. For example, if `word_size=2` then the FSM will be:

            [
                # State 0, which is the root of the Huffman tree
                (state_after_transition, flags, to_emit),
                (state_after_transition, flags, to_emit),
                (state_after_transition, flags, to_emit),
                (state_after_transition, flags, to_emit),

                # State 1
                ...
            ]

        where `flags` is an integer denoting whether `DECODER_FAIL`
        and/or `DECODER_COMPLETE` has occurred.

        Thus indexing into the FSM can be done using:

            curr_state * (1 << word_size) + transition

    """
    if word_size not in [1, 2, 4, 8]:
        raise ValueError("Possible values for word_size are: 1, 2, 4 or 8.")

    # Create all states for the FSM. This can't be done dynamically in
    # the FSM creation algorithm, because transitions could take you to
    # states that aren't visited yet (in a BFS manner).
    state = 0
    q: t.Deque[TreeNode] = deque([root])
    while q:
        node = q.popleft()

        if node.left is not None:
            q.append(node.left)
        if node.right is not None:
            q.append(node.right)

        # In the FSM, only internal Huffman tree nodes will becomes
        # states.
        if node.left is None and node.right is None:
            continue
        else:
            node.fsm_state = state
            state += 1


    def get_individual_bits(num: int) -> t.Generator[t.Literal[0, 1], None, None]:
        """Get individal bits, left to right, from a given number.

        For example, given `num=6` and `word_size=4`, then the binary
        representation would be `0110` and thus the individual bits are
        `0` -> `1` -> `1` -> `0`.

        """
        nonlocal word_size

        for bit_idx in range(word_size-1, -1, -1):
            yield num & (1 << bit_idx)  # type: ignore


    # Generate all transitions for the FSM, where a transition is
    # defined as:
    #   (state_after_transition, is_invalid_transition, to_emit)
    fsm = []
    q: t.Deque[TreeNode] = deque([root])
    num_transitions = 1 << word_size
    while q:
        node = q.popleft()

        if node.left is not None:
            q.append(node.left)
        if node.right is not None:
            q.append(node.right)

        # Only consider internal nodes of the tree.
        if node.left is None and node.right is None:
            continue

        for transition in range(num_transitions):
            node_after_transition = node
            # The character(s) to be emitted for the transition.
            to_emit = ""
            flags = 0

            for bit_value in get_individual_bits(transition):
                if bit_value == Direction.LEFT.value:
                    node_after_transition = node_after_transition.left
                else:
                    node_after_transition = node_after_transition.right

                # The given transition can't be done from the current
                # state. This could happen if the FSM is asked to decode
                # an invalid sequence of bytes, i.e. one that isn't
                # encoded using the given Huffman tree.
                if node_after_transition is None:
                    flags |= DECODER_FAIL
                    break

                # Encountered a leaf node, so transition back to root.
                if (
                    node_after_transition.left is None
                    and node_after_transition.right is None
                ):
                    if node_after_transition.char == PSEUDO_EOF:
                        flags |= DECODER_COMPLETE
                        # Once the PSEUDO_EOF is read, we no longer
                        # decode. If we don't, then `to_emit` could
                        # contain characters that weren't in the
                        # encoded text.
                        break
                    else:
                        to_emit += node_after_transition.char

                    node_after_transition = root

            # Add transition to FSM.
            if flags & DECODER_FAIL:
                fsm.append((None, flags, to_emit))
            else:
                fsm.append(
                    (
                        node_after_transition.fsm_state,  # type: ignore
                        flags,
                        to_emit
                    )
                )

    return fsm


def _encode_freq_table(freq_table: dict[str, int]) -> bytearray:
    # We know this fits into 1 byte, because the frequency table only
    # contains distinct ASCII chars.
    num_chars = len(freq_table)

    # - 1 byte for the number of chars that will follow
    # - 1 byte to contain the char
    # - 4 bytes to contain the count
    freq_table_encoding = struct.pack(
        eval(FREQ_TABLE_ENCODING_SCHEMA.format(num_chars=num_chars)),
        *itertools.chain.from_iterable(
            (c.encode("ASCII"), f) for c, f in freq_table.items()
        ),
    )

    ans = bytearray()
    ans.append(num_chars)
    ans.extend(freq_table_encoding)
    return ans


def _decode_freq_table(encoding: bytes) -> dict[str, int]:
    num_chars = encoding[0]
    decoding = struct.unpack(
        eval(FREQ_TABLE_ENCODING_SCHEMA.format(num_chars=num_chars)),
        encoding[1:],  # First byte is `num_chars`
    )
    freq_table = {
        decoding[i].decode("ASCII"): decoding[i+1]
        for i in range(0, len(decoding), 2)
    }
    return freq_table


# NOTE: 2 pass. So a stream needs to be fed twice or the freq_table
# has to be given first.
# TODO: Even when chunk_size is given, the final byte_encoding is
# kept in memory is well. Ideally, when working with streams the
# byte_encoding is actually the stream we output to immediately.
# TODO: Make f_in optional, if not given then stdin is used
def encode(
    f_in: t.TextIO,
    f_out: t.Optional[t.BinaryIO] = None ,
    chunk_size: t.Optional[int] = None
):
    """Encodes the given text according to the Huffman algorithm.

    Encodes the stream given by `f_in` and writes the encoded stream to
    `f_out`. The encoded text is byte aligned and includes the frequency
    table in the encoding. That way the `decode` function doesn't need
    the original text in order to decode the encoded text.

    When no `chunk_size` is given, then the entire encoding of the text
    is kept in memory. Whilst this is fastest (since no chunking code
    has to run), it might not fit into your memory. Without chunking
    the memory consumption overhead will be equal to the size of `text`.

    Args:
        text: The text to encode.
        chunk_size: Number of bytes to encode at once. This limits the
            total memory consumption of encoding. If `None` is given,
            then no chunking is used, if zero `0` is given, then a
            default chunking size is choosen, otherwise the given size
            is used.

    Returns:
        None. The encoded text will be contained in `f_out`.

    """
    if not f_in.seekable():
        raise TypeError(f"Input stream `f_in` has to be seekable: {f_in}")

    if f_out is None:
        # Use the underlying binary buffer of stdout.
        f_out = sys.stdout.buffer

    # TODO: make chunk_size behave like `buffering` in `io.open()`
    # TODO: Rename chunk_size to buffer_size
    # try:
    #     bs = os.fstat(raw.fileno()).st_blksize
    # except (OSError, AttributeError):
    #     pass
    # else:
    #     if bs > 1:
    #         buffering = bs
    # From here on `chunk_size` will be the number of bits instead of
    # number of bytes, to be encoded at once. That is, because we
    # actually operate on the bit level.
    if chunk_size is not None:
        if chunk_size == 0:
            chunk_size = io.DEFAULT_BUFFER_SIZE  # 8192
        else:
            chunk_size *= 8

    freq_table = _get_freq_table(f_in)
    # Seek start of stream because the frequency table construction
    # has read the entire stream.
    f_in.seek(0)
    # NOTE: Frequency table writing is not chunked.
    f_out.write(_encode_freq_table(freq_table))

    huffman_tree = _get_huffman_tree(freq_table)
    huffman_code = _get_huffman_code(huffman_tree)

    # TODO: https://www.python.org/doc/essays/list2str/
    byte_encoding = bytearray()
    encoding_buffer = []
    bits_in_buffer = 0
    while (chars := f_in.read(io.DEFAULT_BUFFER_SIZE)):
        for char in chars:
            encoded_char = huffman_code[char]
            encoding_buffer.append(encoded_char)

            if chunk_size is None:
                continue

            bits_in_buffer += len(encoded_char)
            if bits_in_buffer < chunk_size:
                continue

            encoding = "".join(encoding_buffer)
            for i in range(0, chunk_size, 8):
                byte = encoding[i:i+8]
                byte_encoding.append(int(byte, 2))

            encoding_buffer = [encoding[chunk_size:]]
            bits_in_buffer = len(encoding) - bits_in_buffer
            encoding = ""  # old string can be GC
            f_out.write(byte_encoding)
            byte_encoding = bytearray()

    encoding_buffer.append(huffman_code[PSEUDO_EOF])
    encoding = "".join(encoding_buffer)
    for i in range(0, len(encoding), 8):
        byte = encoding[i:i+8]
        # Could be that the last byte isn't completely filled and thus
        # we need to make sure it is left-aligned. Otherwise zeros would
        # be inserted into the code (leading to wrong decoding).
        byte_encoding.append(int(byte, 2) << (8 - len(byte)))

    f_out.write(byte_encoding)


# TODO: When working with streams, be careful as the freq_table might
# not fit entirely in the first part of the stream
def decode(f_in: t.BinaryIO, f_out: t.Optional[t.TextIO]) -> None:
    if f_out is None:
        f_out = sys.stdout

    # Get the FSM to use it for decoding.
    # NOTE: byteorder doesn't matter since we are reading just 1 byte.
    num_chars = int.from_bytes(f_in.read(1), byteorder="little")
    # TODO: Seeks are "expensive"!
    f_in.seek(0)
    num_bytes_for_freq_table = 1 + 5*num_chars  # 1 due to num_chars byte
    freq_table = _decode_freq_table(f_in.read(num_bytes_for_freq_table))
    huffman_tree = _get_huffman_tree(freq_table)
    fsm = _get_fsm_decoder(huffman_tree, word_size=DECODER_WORD_SIZE)

    # Dynamically create bit masks based on DECODER_WORD_SIZE. For example,
    # given DECODER_WORD_SIZE=2 then we want to end up with the masks:
    # 11000000, 00110000, 00001100, 00000011
    shifts = list(range(8-DECODER_WORD_SIZE, -1, -DECODER_WORD_SIZE))
    base_mask = 1 << (DECODER_WORD_SIZE - 1)
    base_mask = base_mask ^ (base_mask - 1)
    masks = [base_mask << shift for shift in shifts]

    # Constants.
    num_transitions = 1 << DECODER_WORD_SIZE

    state = 0
    # TODO: Faster to use io.BufferedReader / Writer instead?
    # Don't read just a byte at a time.
    # TODO: use read1 instead? Im so confused. How does buffering
    # work and where is it buffered and why?
    while (bytes := f_in.read(io.DEFAULT_BUFFER_SIZE)):
        for byte in bytes:
            # Feed DECODER_WORD_SIZE number of bits to the FSM.
            for mask, shift in zip(masks, shifts):
                # Index for current state + transition based on given bits.
                idx = (state * num_transitions) + ((byte & mask) >> shift)
                state, flags, emit = fsm[idx]

                if flags & DECODER_FAIL:
                    raise RuntimeError("Invalid code received for given Huffman code.")

                if emit:
                    f_out.write(emit)

                if flags & DECODER_COMPLETE:
                    return

    raise RuntimeError("PSEUDO_EOF was not in the encoding.")


def main():
    # TODO: CLI
    ...

    import io
    import random
    import string

    def get_random_text() -> str:
        text_size = 10**6

        # Get a random text that, just like regular English, has some
        # characters that occur more often than others.
        # NOTE: We know this text doesn't contain the `PSEUDO_EOF`.
        rands = [
            int(random.gauss(50, 15))
            for _ in range(text_size)
        ]
        text = [
            string.printable[r]
            for r in rands
            if 0 <= r <= 99
        ]

        return "".join(text)

    text = get_random_text()
    text_stream = io.StringIO()
    text_stream.write(text)

    byte_encoding = io.BytesIO()
    encode(f_in=text_stream, f_out=byte_encoding)
    byte_encoding.seek(0)

    decoded_text = io.StringIO()
    decode(f_in=byte_encoding, f_out=decoded_text)
    decoded_text.seek(0)


if __name__ == "__main__":
    main()
