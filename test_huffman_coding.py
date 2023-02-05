import os
import random
import string
import unittest

import huffman_coding

encode = huffman_coding.encode
decode = huffman_coding.decode


random.seed(91)


def get_random_text() -> str:
    text_size = 10**6

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


class TestEncodingDecoding(unittest.TestCase):
    test_file_path = "test_byte_encoding.raw"

    def tearDown(self):
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_coding(self):
        """Tests encoding/decoding of random texts."""
        for i in range(10):
            word_size = random.choice([1, 2, 4, 8])
            chunk_size = random.choice([None, 0, 5])
            huffman_coding.DECODER_WORD_SIZE = word_size
            msg = (
                f"Testing whether `text = decode(encode(text))`"
                f" with `DECODER_WORD_SIZE={word_size}` for text: {i}"
            )
            with self.subTest(msg=msg):
                text = get_random_text()
                byte_encoding = encode(text, chunk_size=chunk_size)

                self.assertEqual(text, decode(byte_encoding))

                with open(self.test_file_path, "bw") as f:
                    f.write(byte_encoding)
                with open(self.test_file_path, "br") as f:
                    byte_encoding_from_file = f.read()

                self.assertEqual(byte_encoding, byte_encoding_from_file)


if __name__ == "__main__":
    unittest.main(verbosity=2)
