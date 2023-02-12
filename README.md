2 passes for encoding:
- freq table
- encoding (actually I do this rather dumb so requires another pass)

1 pass decoder using FSM, decoding `DECODER_WORD_SIZE` bits at once.


## Todo

- [ ] CLI
- [ ] Improve performance of encoding
- [ ] Do stdin and stdout work?
- [ ] Benchmark against other Huffman Python implementations on PyPi. See dataset used in Microsoft
  benchmark
    - [ ] Benchmark for different `DECODER_WORD_SIZE`
- [ ] Implement some more performance improvements as per the Microsoft paper (see resources)
- [x] Chunk size name is stange and values as well, 0 is no buffering, > 0 use that value and -1 is
  default which means using recommended.
  [buffering](https://github.com/python/cpython/blob/b652d40f1c88fcd8595cd401513f6b7f8e499471/Lib/_pyio.py#L123)
- [x] Even better to use `os.fstat(f.fileno()).st_blksize` or at least a multiple of it instead of
  the hardcoded `io.DEFAULT_BUFFER_SIZE`
- [x] Working with files that don't fit into memory
- [x] Benchmark using dict vs list for counts. Which has faster lookup?
- [x] Some tests
- [x] There must be a smarter way for the encoding part.
- [x] Working with streams, exclusively! No strings.
    - [x] `get_random_text()` to only return stream
    - [x] Exclusively work with streams, no more strings.
    - [x] Reading just 1 byte at a time is rather slow. Can we do this faster? Maybe read a chunk to
      a buffer (which is then in memory) and then read 1 byte from there each time.
    - [x] I so don't understand buffering... doesn't Python buffer for me and thus reading just 1
      byte is fast? Appears that it is slow.


## Notes

Having to call `seek(0)` on streams after `encode()` and `decode()` calls.

Using `newline=""` on `open()` calls. This can be put in the docstrings of `encode()` en `decode()`
as well.

## Further improvements

- [Embarrassingly parallel Huffman encoding](http://www.ittc.ku.edu/~jsv/Papers/HoV95.pdcfull.pdf)

## Resources

-   [Explanation of Huffman Encoding - Stanford course](https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1126/handouts/220%20Huffman%20Encoding.pdf)
-   Idea of using [a static FSM decoding table for HTTP headers](https://github.com/python-hyper/hpack/blob/master/src/hpack/huffman_table.py) ([RFC standard for HPACK](https://www.rfc-editor.org/rfc/rfc7541#appendix-B))
- [Data-Parallel Finite-State Machines by Microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/asplos302-mytkowicz.pdf)
