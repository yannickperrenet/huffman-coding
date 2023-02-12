import io
import os
import random
import string


def get_random_text() -> str:
    text_size = 10**7

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

def loop_byte(stream):
    while (byte := stream.read(1)):
        ...

def loop_buffered(stream):
    buffering = io.DEFAULT_BUFFER_SIZE
    # try:
    #     bs = os.fstat(stream.fileno()).st_blksize
    # except (OSError, AttributeError):
    #     pass
    # else:
    #     if bs > 1:
    #         buffering = bs

    while (bytes := stream.read(buffering)):
        for byte in bytes:
            ...

def loop_buffered_generator(stream):
    while (bytes := stream.read(io.DEFAULT_BUFFER_SIZE)):
        for byte in bytes:
            yield byte

def main():
    # import gc
    # gc.disable()

    # text = get_random_text()
    # text_bytes = text.encode("ascii")

    # # BytesIO is a stream of in-memory bytes and is buffered.
    # # https://github.com/python/cpython/blob/b652d40f1c88fcd8595cd401513f6b7f8e499471/Lib/_pyio.py
    # # https://github.com/python/cpython/blob/b652d40f1c88fcd8595cd401513f6b7f8e499471/Modules/_io/bytesio.c#L392
    # # https://github.com/python/cpython/blob/b652d40f1c88fcd8595cd401513f6b7f8e499471/Lib/io.py
    # text_stream = io.BytesIO()
    # # For every read on `io.BytesIO`:
    # # Read X from underlying bytearray()
    # # Convert read byte(s) to `bytes(...)`
    # # (bytes() is immutable bytearray())
    # # ----
    # # I am guessing that os.open(buffering=...) will mean what io class
    # # is returned. And thus whether just 1 byte is read from the file
    # # or whether it is served from an in-memory buffer.
    # # https://github.com/python/cpython/blob/b652d40f1c88fcd8595cd401513f6b7f8e499471/Lib/_pyio.py#L247
    # text_stream.write(text_bytes)

    ####################
    # To read just 1 byte from file stream or chunk
    ####################
    # Interesting test because reading a text file will result in a
    # `io.BufferedReader`. Thus is it faster to not rely on the
    # buffering of the `io.BufferedReader`?

    # with open("test_output.txt", "w") as f:
    #     f.write(text)

    # with open("test_output.txt", "r") as f:
    #     for _ in range(50):
    #         f.seek(0)
    #         loop_byte(f)

    # NOTE: This is about 5x faster.
    # Took around 5s -> 100mb/s. Why so slow?
    with open("test_output.txt", "r") as f:
        for _ in range(50):
            f.seek(0)
            loop_buffered(f)

            # NOTE: This takes the same time.
            # Just doing `f.read()` is about 10x faster
            # content = f.read()
            # for char in content:
            #     ...


    ####################
    # To read just 1 byte from stream or chunk?
    ####################

    # for _ in range(100):
    #     text_stream.seek(0)
    #     loop_byte(text_stream)

    # NOTE: This is about 4x faster.
    # Thus using a `io.BufferedReader` wouldn't help since reading from
    # that is pretty much identical to reading from `io.BytesIO`. Both
    # would try to serve from a `bytearray()` (as the buffer).
    # for _ in range(100):
    #     text_stream.seek(0)
    #     loop_buffered(text_stream)

    ####################
    # To generator or not?
    ####################

    # for _ in range(100):
    #     text_stream.seek(0)
    #     gen = loop_buffered_generator(text_stream)
    #     for byte in gen:
    #         ...

    # NOTE: This is about 4x faster.
    # for _ in range(100):
    #     text_stream.seek(0)
    #     while (bytes := text_stream.read(io.DEFAULT_BUFFER_SIZE)):
    #         for byte in bytes:
    #             ...


if __name__ == "__main__":
    main()
