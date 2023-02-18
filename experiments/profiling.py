"""

```sh
python -m cProfile -o perf_test.prof huffman_coding.py
snakeviz perf_test.prof
```

"""
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
    N = 10**5  # ~100MB
    write_large_file(N=N)

    print("Encoding large file.")
    with (
        open(FILE_PATH_TEXT, mode="r", newline="") as f_in,
        open(FILE_PATH_BYTES, mode="wb", buffering=0) as f_out
    ):
        encode(f_in=f_in, f_out=f_out)

    print("Decoding large file.")
    with (
        open(FILE_PATH_BYTES, mode="rb", buffering=0) as f_in,
        open(FILE_PATH_TEXT, mode="w", newline="") as f_out
    ):
        decode(f_in=f_in, f_out=f_out)
