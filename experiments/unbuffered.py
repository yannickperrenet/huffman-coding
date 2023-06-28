import os


FILE_PATH_BYTES = "larger_than_memory_file.raw"
FILE_PATH_BYTES_OUT = "larger_than_memory_file_out.raw"


def write_large_file(N):
    with open(FILE_PATH_BYTES, "wb") as f:
        for _ in range(N):
            f.write((1000 * "a").encode("ascii"))


if __name__ == "__main__":
    print("Write large file.")
    # For N=10**5 = 100MB
    # For chunk_size=None -- Peak: 948408995 = 1GB
    # For chunk_size=0 -- Peak: 108248 = 0.1MB
    N = 10**3  # ~1MB
    write_large_file(N=N)

    # Question: What if we try to write smaller than block size
    # to unbuffered file?
    # A: My guess is that the OS will keep it in memory and thus
    # it is fine.
    with (
        open(FILE_PATH_BYTES, mode="rb") as f_in,
        open(FILE_PATH_BYTES_OUT, mode="wb", buffering=0) as f_out
    ):
        while (byte := f_in.read(1)):
            f_out.write(byte)
            f_out.flush()
            # os.fsync(f_out.fileno())
