import io
import os
import random
import string
import unittest

import huffman_coding


random.seed(91)

encode = huffman_coding.encode
decode = huffman_coding.decode


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


class TestEncodingDecoding(unittest.TestCase):
    file_path_text = "test_text.txt"
    file_path_bytes = "test_bytes.raw"

    def tearDown(self):
        if os.path.exists(self.file_path_text):
            os.remove(self.file_path_text)
        if os.path.exists(self.file_path_bytes):
            os.remove(self.file_path_bytes)

    # TODO: Another test that uses file stream.
    def test_coding(self):
        """Tests encoding/decoding of random texts."""
        text = get_random_text()

        parametrizations = get_parametrizations(
            [1, 2, 4, 8],  # word_size
            [None, 0, 5],  # chunk_size
        )
        for parametrization in parametrizations:
            word_size, chunk_size = parametrization
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
                encode(f_in=text_stream, f_out=byte_encoding, chunk_size=chunk_size)
                byte_encoding.seek(0)

                decoded_text = io.StringIO()
                decode(f_in=byte_encoding, f_out=decoded_text)
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
                    encode(f_in=f_in, f_out=f_out, chunk_size=chunk_size)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
