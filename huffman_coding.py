"""

- https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1126/handouts/220%20Huffman%20Encoding.pdf
- http://www.cs.bc.edu/~signoril/cs102/projects/fall2007/project7.html
- https://rosettacode.org/wiki/Huffman_coding#Python

Static huffman code table that uses the "optimal" static code for HTTP
headers (optimal code on a large corpus of example headers):

- https://www.rfc-editor.org/rfc/rfc7541#appendix-B
- https://github.com/python-hyper/hpack/blob/master/src/hpack/huffman_table.py

Creating a Transducer (FSM) for codes:

- https://hal.science/hal-00620817/document (just above example 1.15)

Unrolling an FSM:

- https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/asplos302-mytkowicz.pdf

"""
import heapq
import pprint
import typing as t
from collections import defaultdict, deque
from enum import Enum


# TODO: Choose dynamically based on input.
# This is to ensure we stop decoding instead of having to pad the
# output.
PSEUDO_EOF = chr(4)  # End Of Transmission

# TODO: Can't we actually take values like 16 as well? Just need to
# loop differently through the bytes when decoding, I guess.
# Number of bits to process at once using a Finite State Machine (FSM).
DECODER_WORD_SIZE = 4


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


def _get_freq_table(text: str) -> dict[str, int]:
    freq_table = defaultdict(int)
    for char in text:
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


# TODO: Don't require text to be able to fit into memory. Work with
# streams instead.
def encode(text: str) -> bytearray:
    freq_table = _get_freq_table(text)
    huffman_tree = _get_huffman_tree(freq_table)
    huffman_code = _get_huffman_code(huffman_tree)

    # TODO: store freq table as well.
    encoding = []
    for char in text:
        encoding.append(huffman_code[char])
    encoding.append(huffman_code[PSEUDO_EOF])
    encoding = "".join(encoding)

    byte_encoding = bytearray()
    for i in range(8, len(encoding), 8):
        byte = encoding[i-8:i]
        byte_encoding.append(int(byte, 2))

    return byte_encoding


def _get_fsm_decoder(root: TreeNode, word_size: int = DECODER_WORD_SIZE) -> list:
    """Gets an FSM that can be used as a Huffman decoder.

    Each internal node of the Huffman tree (including the root node)
    will become a state in the FSM. For each state we consider all
    possible transitions based on the specified `word_size`. For
    example, when `word_size=4` the possible transitions are the bit
    sequences: `00`, `01`, `10` and `11`.

    TODO:
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
            (state_after_transition, is_invalid_transition, to_emit),
            (state_after_transition, is_invalid_transition, to_emit),
            (state_after_transition, is_invalid_transition, to_emit),
            (state_after_transition, is_invalid_transition, to_emit),

            # State 1
            ...
        ]

        Thus indexing into the FSM can be done using:

            word_size*curr_state + transition

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
            is_invalid_transition = False
            # The character(s) to be emitted for the transition.
            to_emit = ""

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
                    is_invalid_transition = True
                    break

                # Encountered a leaf node, so transition back to root.
                if (
                    node_after_transition.left is None
                    and node_after_transition.right is None
                ):
                    to_emit += node_after_transition.char
                    node_after_transition = root

            # Add transition to FSM.
            if is_invalid_transition:
                fsm.append([None, is_invalid_transition, to_emit])
            else:
                fsm.append(
                    [
                        node_after_transition.fsm_state,  # type: ignore
                        is_invalid_transition,
                        to_emit
                    ]
                )

    return fsm


# TODO: work with bytearray instead? Or how do I make it work for
# large streams?
def decode(code: bytes, fsm: list) -> str:
    # TODO: Possibly just hardcoding is clearer.
    # Dynamically create bit masks based on DECODER_WORD_SIZE. For example,
    # given DECODER_WORD_SIZE=2 then we want to end up with the masks
    # 11000000, 00110000, 00001100, 00000011
    shifts = list(range(8-DECODER_WORD_SIZE, -1, -DECODER_WORD_SIZE))
    base_mask = 2**(DECODER_WORD_SIZE - 1)
    base_mask = base_mask ^ (base_mask - 1)
    masks = [base_mask << shift for shift in shifts]

    # TODO: These print lines are actually pretty usefull for
    # debugging. Maybe allow a DEBUG = True param?. Convert to logging
    # module probably.
    ans = []
    state = 0
    for byte in code:
        # print()
        # print(byte, byte.to_bytes(1, "little"), bin(byte)[2:].rjust(8, "0"))

        # Feed DECODER_WORD_SIZE number of bits to the FSM.
        for mask, shift in zip(masks, shifts):
            idx = (state * 2**DECODER_WORD_SIZE) + ((byte & mask) >> shift)
            state, invalid, emit = fsm[idx]
            # print((state, idx), end='->')

            if invalid:
                raise RuntimeError("Invalid code received for given FSM.")

            if emit:
                if PSEUDO_EOF in emit:
                    return "".join(ans)
                ans.append(emit)

    return "".join(ans)


def main():
    with open("file_text.txt", "r") as f:
        text = f.read()
    with open("hamlet.txt", "r") as f:
        text = f.read()

    freq_table = _get_freq_table(text)
    huffman_tree = _get_huffman_tree(freq_table)

    # TODO: get fsm to process bytes for decoding
    fsm = _get_fsm_decoder(huffman_tree, word_size=DECODER_WORD_SIZE)

    byte_encoding = encode(text)

    # print(text)
    # print(encoding)
    # print(len(encoding) % 8)
    # print()
    # print(byte_encoding)
    # print()
    # print(huffman_code)
    # print()
    # pprint.pprint(fsm)
    # print()

    with open("file_binary.raw", "bw") as f:
        f.write(byte_encoding)

    with open("file_binary.raw", "br") as f:
        byte_encoding_from_file = f.read()

    # TODO: Actually... the freq table needs to be included in the
    # encoding as well.
    # - 1 byte for the number of chars that will follow
    # - 1 byte to contain the char
    # - 4 bytes to contain the count
    output = decode(code=byte_encoding, fsm=fsm)
    assert output == text
    output = decode(code=byte_encoding_from_file, fsm=fsm)
    assert output == text
    # print(output)


def run_tests():
    text = "AABACDACA"
    freq_table = _get_freq_table(text)
    huffman_tree = _get_huffman_tree(freq_table)
    huffman_code = _get_huffman_code(huffman_tree)
    answer = {'B': '111', 'D': '110', 'C': '10', 'A': '0'}
    assert huffman_code == answer

    # This was without the PSEUDO_EOF
    text = "aaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbcccccccccccccccccccddddddddddddddddddddcba\n"
    fsm = [
        [0, 'd'], [0, 'a'], [0, 'c'], [3, ''],
        [1, 'd'], [2, 'd'], [1, 'a'], [2, 'a'],
        [1, 'c'], [2, 'c'], [0, '\n'], [0, 'b'],
        [1, '\n'], [2, '\n'], [1, 'b'], [2, 'b']
    ]
    assert _get_fsm_decoder(_get_huffman_tree(text), word_size=2) == fsm



if __name__ == "__main__":
    main()
