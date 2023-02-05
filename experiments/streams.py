import io
import sys

"""CLI.

Possible inputs
- path to a file

Only allows "path to a file" as input because most streams aren't
seekable (which is required since `encode()` needs two passes). Thus the
solution to work with those is to write to a file and then read from
that file again.

>>> sys.stdin.seekable()
False

Text I/O is needed for input because that will contain "regular" text.


From Python docs ([source](https://docs.python.org/3/library/io.html#id2)):

    By reading and writing only large chunks of data even when the user
    asks for a single byte, buffered I/O hides any inefficiency in
    calling and executing the operating system’s unbuffered I/O
    routines. The gain depends on the OS and the kind of I/O which is
    performed. For example, on some modern OSes such as Linux,
    unbuffered disk I/O can be as fast as buffered I/O. The bottom line,
    however, is that buffered I/O offers predictable performance
    regardless of the platform and the backing device. Therefore, it is
    almost always preferable to use buffered I/O rather than unbuffered
    I/O for binary data.


./huffman_coding encode [INPUT_FILE] [OUTPUT_STREAM]
- if [OUTPUT STREAM] is not given, then defaults to sys.stdout

./huffman_coding decode [INPUT STREAM] [OUTPUT_STREAM]
- Stream doesn't need to be seekable since decode needs 1 pass


with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as stdout:
    stdout.write(b"my bytes object")
    stdout.flush()

"""

import typing as t
import os

def encode(fin: t.Union[t.TextIO, str], fout: t.Optional[t.BinaryIO] = None) -> None:
    if not isinstance(fin, str) and not fin.seekable():
        raise TypeError(f"Input stream `fin` has to be seekable: {fin}")

    # NOTE: Should be fast as Python automatically buffers
    # for char in fin:
        # get freq table
    for char in fin:
        print(char, end="")
    if not isinstance(fin, str):
        fin.seek(0)
    # Now encode when reading from `fin` and write to `fout`
    if fout is None:
        # Use the underlying binary buffer of stdout.
        fout = sys.stdout.buffer
        # fout = os.fdopen(sys.stdout.fileno(), "wb", closefd=False)
    for char in fin:
        fout.write(char.encode("utf-8"))


def decode(fin: t.BinaryIO, fout: t.Optional[t.TextIO]) -> None:
    ...


def main():
    # Buffered reading and writing. Just let the system do its thing
    # as we don't need unbuffered writing to ensure it is written. We
    # just care about performance.
    with (
        open("hamlet.txt", mode="r", encoding="utf-8") as fin,
        open("encoded_hamlet.raw", mode="wb") as fout
    ):
        # encode(fin, fout)
        encode(fin)
        # encode(sys.stdin)

    encode("hi there\n")


if __name__ == "__main__":
    main()
