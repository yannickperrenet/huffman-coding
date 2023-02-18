# Huffman coding

[Huffman coding](https://en.wikipedia.org/wiki/Huffman_coding) Python implementation.

Why another implementation you ask?

-   It is a pure Python implementation with no additional dependencies outside the standard library.
-   It works with streams exclusively, in a buffered fashion. Meaning that you can encode and decode
    files (or streams) that are larger than memory. The total memory consumption peaks at just
    `70KB`.
-   In contrast to other implementations, the encoding step actually encodes the frequency table as
    well (which the decode step expects to read). This means that the original text does not need to
    be kept for decoding.

    Let's say you want to share a file over network, then it often improves performance to first
    compress the file before sending it over the wire. However, you want to be able to
    decompress/decode the received file without requiring the original (since that would destroy the
    entire purpose of compressing in the first place).

-   The decoding step uses an [FSM](https://en.wikipedia.org/wiki/Finite-state_machine) so it can
    operate on an entire byte at once, instead of decoding bit-by-bit. This means the decoding step
    is significantly faster (`4x`).

    This is not a novel idea though, although simply not implemented by other pacakges. Basically, I
    noticed that the RFC for [HPACK: Header Compression for
    HTTP/2](https://www.rfc-editor.org/rfc/rfc7541#appendix-B) included a static Huffman code which
    was being decoded using an FSM in [Python's HPACK
    implementation](https://github.com/python-hyper/hpack/blob/v4.0.0/src/hpack/huffman_table.py#L131)
    which essentially "unrolled" the FSM by operating on a byte level instead of bit level. I simply
    had to generate a FSM dynamically based on the Huffman code (see `_get_fsm_decoder()`) to be
    able to apply the same principle to non-static Huffman codes.


## Usage

Let's use Hamlet as a test text.
```sh
git clone https://github.com/yannickperrenet/huffman-coding.git && cd huffman-coding
wget https://gist.githubusercontent.com/provpup/2fc41686eab7400b796b/raw/b575bd01a58494dfddc1d6429ef0167e709abf9b/hamlet.txt -O hamlet.txt
```

### CLI
```sh
# Encode a given file and write to an output file.
./huffman_coding.py encode --input="hamlet.txt" --output="hamlet.raw"
# Decode from file and write to STDOUT.
./huffman_coding.py decode --input="hamlet.raw"
# Encode file and decode immediately again.
./huffman_coding.py encode --input="hamlet.txt" | ./huffman_coding.py decode
```

### Python

#### Files

```python
# NOTE: Make sure to pass `newline=""` to `open()` to prevent newline
# translation since that messes with encoding/decoding. Read more here:
# https://docs.python.org/3/library/io.html#io.TextIOWrapper
# NOTE: Disable buffering because `encode()` and `decode()` already
# work in a buffered fashion.
with (
    open("hamlet.txt", mode="r", newline="") as f_in,
    open("hamlet.raw", mode="wb", buffering=0) as f_out
):
    encode(f_in=f_in, f_out=f_out)

with (
    open("hamlet.raw", mode="rb", buffering=0) as f_in,
    open("hamlet.txt", mode="w", newline="") as f_out
):
    decode(f_in=f_in, f_out=f_out)
```

#### In-memory

```python
import io

with open("hamlet.txt", mode="r", newline="") as f:
    text = f.read()

text_stream = io.StringIO()
text_stream.write(text)
# Seek start of stream, otherwise successive reads won't return
# anything.
text_stream.seek(0)

byte_encoding = io.BytesIO()
encode(f_in=text_stream, f_out=byte_encoding, buffering=buffering)
byte_encoding.seek(0)

decoded_text = io.StringIO()
decode(f_in=byte_encoding, f_out=decoded_text)
decoded_text.seek(0)
```

## Tests

Simply run `python3 test_huffman_coding.py`.

## Resources

-   [Embarrassingly parallel Huffman encoding](http://www.ittc.ku.edu/~jsv/Papers/HoV95.pdcfull.pdf)
-   [Explanation of Huffman Encoding - Stanford
    course](https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1126/handouts/220%20Huffman%20Encoding.pdf)
-   [Data-Parallel Finite-State Machines by
    Microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/asplos302-mytkowicz.pdf)
