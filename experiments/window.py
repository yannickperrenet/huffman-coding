import collections
from itertools import islice


def sliding_window(iterable, n):
    # sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = collections.deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)

def get_bytes(encoding):
    stop = len(encoding) - len(encoding) % 8

    for i in range(0, stop, 8):
        byte = encoding[i:i+8]
        yield int(byte, 2)

if __name__ == "__main__":
    encoding = "01110111"
    ans = list(sliding_window(encoding, 2))
    print(ans)
    print(list(get_bytes(encoding)))
