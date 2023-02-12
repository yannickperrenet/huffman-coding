import io
import os
import random
import string
import unittest

import huffman_coding


random.seed(91)

encode = huffman_coding.encode
decode = huffman_coding.decode


def get_random_text() -> str:
    text_size = 10**4

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
    test_file_path = "test_byte_encoding.raw"

    def tearDown(self):
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    # TODO: Another test that uses file stream.
    def test_coding(self):
        """Tests encoding/decoding of random texts."""
        text = get_random_text()
        text_stream = io.StringIO()
        text_stream.write(text)

        parametrizations = get_parametrizations(
            [1, 2, 4, 8],  # word_size
            [None, 0, 5],  # chunk_size
        )
        for parametrization in parametrizations:
            text_stream.seek(0)

            word_size, chunk_size = parametrization
            huffman_coding.DECODER_WORD_SIZE = word_size

            msg = (
                f"Testing paramatrization: {parametrization}"
            )
            with self.subTest(msg=msg):
                byte_encoding = io.BytesIO()
                encode(f_in=text_stream, f_out=byte_encoding, chunk_size=chunk_size)
                byte_encoding.seek(0)

                decoded_text = io.StringIO()
                decode(f_in=byte_encoding, f_out=decoded_text)
                decoded_text.seek(0)
                text_stream.seek(0)
                self.assertEqual(text_stream.read(), decoded_text.read())

                # with open(self.test_file_path, "bw") as f:
                #     f.write(byte_encoding)
                # with open(self.test_file_path, "br") as f:
                #     byte_encoding_from_file = f.read()

                # self.assertEqual(byte_encoding, byte_encoding_from_file)


if __name__ == "__main__":
    unittest.main(verbosity=2)
