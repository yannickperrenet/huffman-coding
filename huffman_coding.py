"""

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


WORD_SIZE = 2


class Direction(Enum):
    LEFT = 0
    RIGHT = 1


class TreeNode:
    def __init__(
        self,
        char: str = "",
        left: t.Optional["TreeNode"] = None,
        right: t.Optional["TreeNode"] = None,
    ) -> None:
        self.char = char
        self.left = left
        self.right = right
        self.idx = None

    def assign_index(self, idx: int) -> None:
        self.idx = idx

    def __lt__(self, _):
        # No ordering, just default to self.
        return self

    def __str__(self):
        return f"TreeNode<{self.char}>"

    def __repr__(self):
        return self.__str__()


# TODO: coding should be bytes not string
def get_huffman_tree(text: str) -> TreeNode:
    # counts
    counts = defaultdict(int)
    for char in text:
        counts[char] += 1

    # create TreeNode per character
    # sort TreeNode list by counts
    # Use heap
    heap = []
    for char in counts:
        node = TreeNode(char=char)
        # TODO: namedtuple
        heap.append((counts[char], node))

    # Build Tree
    heapq.heapify(heap)
    while len(heap) != 1:
        smallest1, smallest2 = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(
            heap,
            (
                smallest1[0] + smallest2[0],  # count
                TreeNode(
                    left=smallest1[1],
                    right=smallest2[1],
                ),
            ),
        )

    # Don't lose root node
    return heap[0][1]



# TODO: make left/right 0/1 config param.
# Use graph traversal to create code
def construct_code(root: TreeNode) -> dict[str, str]:
    def helper(root: TreeNode) -> None:
        nonlocal ans, codeword

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


def get_huffman_coding(text: str) -> dict[str, str]:
    return construct_code(get_huffman_tree(text))


def get_fsm(root: TreeNode, word_size: int = WORD_SIZE) -> list:
    if word_size not in [1, 2, 4]:
        raise ValueError("Possible values for word_size are: 1, 2 or 4.")

    # Assign index to each node in the Tree. Need to do it before
    # processing because a state transition could take you to a state
    # that wasn't seen before (when going top-down in the tree).
    idx = 0
    q: t.Deque[t.Optional[TreeNode]] = deque([root])
    while q:
        node = q.popleft()
        if node is None:
            continue
        if node.left is None and node.right is None:
            continue

        node.assign_index(idx)
        idx += 1
        q.extend([node.left, node.right])


    bits_to_explore = [2**i for i in range(word_size-1, -1, -1)]
    # Create FSM.
    fsm = []
    q2: t.Deque[TreeNode] = deque([root])
    while q2:
        state = q2.popleft()

        # Only consider internal nodes of the tree.
        if state.left is None and state.right is None:
            continue

        if state.left is not None:
            q2.append(state.left)
        if state.right is not None:
            q2.append(state.right)

        for transition in range(2**word_size):
            new_state = state
            emit = ""
            invalid = False

            # Explore first, second, third, fourth bit
            for mask in bits_to_explore:
                bit = transition & mask

                if bit == Direction.LEFT.value:
                    new_state = new_state.left
                else:
                    new_state = new_state.right

                # Invalid code.
                # The given code said to transition to the right, but
                # there is no tree defined. Thus we have received an
                # invalid code for this tree.
                if new_state is None:
                    invalid = True
                    break

                # Encountered a leaf node, so transition back to root.
                if new_state.left is None and new_state.right is None:
                    emit += new_state.char
                    new_state = root

            # Add transition to FSM.
            # [state transitioned to, invalid, chars to emit]
            # Invalid means that the given transition wasn't valid for
            # the code.
            if invalid:
                fsm.append([None, invalid, emit])
            else:
                fsm.append([new_state.idx, invalid, emit])  # type: ignore

    return fsm


def run_tests():
    text = "AABACDACA"
    huffman_code = get_huffman_coding(text)
    answer = {'B': '111', 'D': '110', 'C': '10', 'A': '0'}
    assert huffman_code == answer

    text = "aaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbcccccccccccccccccccddddddddddddddddddddcba\n"
    fsm = [
        [0, 'd'], [0, 'a'], [0, 'c'], [3, ''],
        [1, 'd'], [2, 'd'], [1, 'a'], [2, 'a'],
        [1, 'c'], [2, 'c'], [0, '\n'], [0, 'b'],
        [1, '\n'], [2, '\n'], [1, 'b'], [2, 'b']
    ]
    assert get_fsm(get_huffman_tree(text), word_size=2) == fsm


# TODO: work with bytearray instead? Or how do I make it work for
# large streams?
def decode(code: bytes, fsm: list) -> str:
    state = 0

    # ans = []
    # for byte in code:
    #     idx = (state * 2**WORD_SIZE) + byte >> 4
    #     print(idx)
    #     state, invalid, emit = fsm[idx]

    #     if invalid:
    #         raise RuntimeError("Invalid code received for given FSM.")

    #     if emit:
    #         ans.append(emit)

    #     idx = (state * 2**WORD_SIZE) + byte & 0x0F
    #     state, invalid, emit = fsm[idx]

    #     if invalid:
    #         raise RuntimeError("Invalid code received for given FSM.")

    #     if emit:
    #         ans.append(emit)
    ans = []
    for byte in code:
        print(byte)
        for mask, shift in zip([3 << 6, 3 << 4, 3 << 2, 3], [6, 4, 2, 0]):
            idx = (state * 2**WORD_SIZE) + (byte & mask) >> shift
            print(idx)
            state, invalid, emit = fsm[idx]

            if invalid:
                raise RuntimeError("Invalid code received for given FSM.")

            if emit:
                ans.append(emit)

    return "".join(ans)


def main():
    with open("file_text.txt", "r") as f:
        text = f.read()

    huffman_tree = get_huffman_tree(text)

    # TODO: get fsm to process bytes for decoding
    fsm = get_fsm(huffman_tree, word_size=WORD_SIZE)
    pprint.pprint(fsm)

    huffman_code = construct_code(huffman_tree)
    encoding = []
    for char in text:
        encoding.append(huffman_code[char])
    encoding = "".join(encoding)

    # To be fair, the huffman code would have to be stored as well. But
    # this becomes negligible as the initial file contains more text,
    # thus we leave it out.
    print(text)
    print(huffman_code)
    print(encoding)

    # TODO: Deal with proper binary encoding.
    # NOTE: Important that a leading 0 is also included, or we can't
    # decode.
    # Solution: Pad with "1"s at the left side of the encoding so that
    # you end up with a multiple of 8 for the total encoding. Then add
    # this padding integer to the front again so that you know what
    # padding was used when decoding the file and can strip that of
    # the encoding.
    length = len(encoding) // 8 + 1  # math.ceil
    # TODO: WRONG! Inserting zeros at the front will lead to wrong
    # decoding.
    output = int(encoding, 2).to_bytes(length, "big")
    print(output)
    with open("file_binary.raw", "ba") as f:
        # TODO: Use struct module instead. Better to chunk into smaller
        # integers then using 1 very large one (not feasible for long
        # texts).
        f.write(output)

    output = decode(code=output, fsm=fsm)
    print(output)


if __name__ == "__main__":
    main()
