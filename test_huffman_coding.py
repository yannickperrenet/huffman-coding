import io
import os
import random
import string
import tracemalloc
import unittest

import huffman_coding


random.seed(91)

encode = huffman_coding.encode
decode = huffman_coding.decode


class TestEncodingDecoding(unittest.TestCase):
    file_path_text = "test_text.txt"
    file_path_bytes = "test_bytes.raw"

    def tearDown(self):
        if os.path.exists(self.file_path_text):
            os.remove(self.file_path_text)
        if os.path.exists(self.file_path_bytes):
            os.remove(self.file_path_bytes)

    def test_coding(self):
        """Tests encoding/decoding of random texts."""
        text = get_random_text()

        parametrizations = get_parametrizations(
            [1, 2, 4, 8],  # word_size
            [-1, io.DEFAULT_BUFFER_SIZE*10],  # buffering
        )
        for parametrization in parametrizations:
            word_size, buffering = parametrization
            huffman_coding.DECODER_WORD_SIZE = word_size

            msg = (
                f"Testing paramatrization: {parametrization}"
            )
            with self.subTest(msg=msg):
                # In-memory streams.
                text_stream = io.StringIO()
                text_stream.write(text)
                text_stream.seek(0)

                byte_encoding = io.BytesIO()
                encode(f_in=text_stream, f_out=byte_encoding, buffering=buffering)
                byte_encoding.seek(0)

                decoded_text = io.StringIO()
                decode(f_in=byte_encoding, f_out=decoded_text, buffering=buffering)
                decoded_text.seek(0)
                self.assertEqual(text, decoded_text.read())

                # File streams.
                # NOTE: Use `newline=""` to make sure no newline
                # translation is used. See:
                # https://docs.python.org/3/library/io.html#io.TextIOWrapper
                with open(self.file_path_text, "w", newline="") as f:
                    f.write(text)

                with (
                    open(self.file_path_text, mode="r", newline="") as f_in,
                    open(self.file_path_bytes, mode="wb") as f_out
                ):
                    encode(f_in=f_in, f_out=f_out, buffering=buffering)

                # Just delete the file to make sure we don't overwrite.
                if os.path.exists(self.file_path_text):
                    os.remove(self.file_path_text)

                with (
                    open(self.file_path_bytes, mode="rb") as f_in,
                    open(self.file_path_text, mode="w", newline="") as f_out
                ):
                    decode(f_in=f_in, f_out=f_out)

                with (
                    open(self.file_path_text, mode="r", newline="") as f,
                ):
                    self.assertEqual(text, f.read())

    def test_memory_consumption(self):
        """Tests memory consumption.

        To be able to work with larger than memory files, the algorithms
        need to work in a buffered fashion. In other words, the memory
        consumption of the program should stay constant and not scale
        with the size of the input.

        """
        # Size can be even larger, but that will make the test take longer.
        write_large_file(self.file_path_text, size=int(5e6))  # ~5MB

        tracemalloc.start()

        with (
            open(self.file_path_text, mode="r", newline="") as f_in,
            open(self.file_path_bytes, mode="wb", buffering=0) as f_out
        ):
            encode(f_in=f_in, f_out=f_out)

        # Just delete the file to make sure we don't overwrite.
        os.remove(self.file_path_text)

        # NOTE: Disable buffering when reading bytes because `decode()`
        # already buffers reading.
        with (
            open(self.file_path_bytes, mode="rb", buffering=0) as f_in,
            open(self.file_path_text, mode="w", newline="") as f_out
        ):
            decode(f_in=f_in, f_out=f_out)

        _, peak_consumption = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertLess(peak_consumption, 2e5)  # 200KB


def get_random_text(text_size=10**4) -> str:
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


def write_large_file(file_path, size=10**8) -> None:
    with open(file_path, "w") as f:
        for _ in range(size//1000):
            f.write(1000 * "a")


def get_parametrizations(*iterables: list) -> list:
    def helper(i: int) -> None:
        if i == N:
            ans.append(parametrization[:])
            return

        for elt in iterables[i]:
            parametrization.append(elt)
            helper(i+1)
            parametrization.pop()

    if not iterables:
        return []

    N = len(iterables)
    ans = []
    parametrization = []
    helper(0)
    return ans


if __name__ == "__main__":
    unittest.main(verbosity=2)
