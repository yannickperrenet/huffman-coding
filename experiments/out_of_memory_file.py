"""Testing whether memory consumption is not file size dependent.

We want the ability to deal with files larger than memory.

"""

import os
import tracemalloc

import huffman_coding

encode = huffman_coding.encode
decode = huffman_coding.decode

FILE_PATH_TEXT = "larger_than_memory_file.txt"
FILE_PATH_BYTES = "larger_than_memory_file.raw"


def write_large_file(N):
    with open(FILE_PATH_TEXT, "w") as f:
        for _ in range(N):
            f.write(1000 * "a")


if __name__ == "__main__":
    print("Write large file.")
    # For N=10**5 = 100MB
    # For chunk_size=None -- Peak: 948408995 = 1GB
    # For chunk_size=0 -- Peak: 108248 = 0.1MB
    N = 10**5  # ~100MB
    write_large_file(N=N)

    tracemalloc.start()

    print("Encoding large file.")
    with (
        open(FILE_PATH_TEXT, mode="r", newline="") as f_in,
        open(FILE_PATH_BYTES, mode="wb") as f_out
    ):
        encode(f_in=f_in, f_out=f_out, chunk_size=0)

    # Just delete the file to make sure we don't overwrite.
    os.remove(FILE_PATH_TEXT)

    print("Decoding large file.")
    with (
        open(FILE_PATH_BYTES, mode="rb") as f_in,
        open(FILE_PATH_TEXT, mode="w", newline="") as f_out
    ):
        decode(f_in=f_in, f_out=f_out)

    traced_memory = tracemalloc.get_traced_memory()
    print(f"Current: {traced_memory[0]}, Peak: {traced_memory[1]}")
    tracemalloc.stop()
