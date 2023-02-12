2 passes for encoding:
- freq table
- encoding (actually I do this rather dumb so requires another pass)

1 pass decoder using FSM, decoding `DECODER_WORD_SIZE` bits at once.


## Todo

- [x] There must be a smarter way for the encoding part.
- [ ] Even better to use `os.fstat(f.fileno()).st_blksize` or at least a multiple of it instead of
  the hardcoded `io.DEFAULT_BUFFER_SIZE`
- [ ] Working with streams, exclusively! No strings.
    - [ ] `get_random_text()` to only return stream
    - [ ] Exclusively work with streams, no more strings.
    - [x] Reading just 1 byte at a time is rather slow. Can we do this faster? Maybe read a chunk to
      a buffer (which is then in memory) and then read 1 byte from there each time.
    - [x] I so don't understand buffering... doesn't Python buffer for me and thus reading just 1
      byte is fast? Appears that it is slow.
- [ ] Benchmark using dict vs list for counts. Which has faster lookup?
- [ ] Working with files that don't fit into memory
- [ ] Benchmark against other Huffman Python implementations on PyPi. See dataset used in Microsoft
  benchmark
    - [ ] Benchmark for different `DECODER_WORD_SIZE`
- [x] Some tests
- [ ] Implement some more performance improvements as per the Microsoft paper (see resources)


## Notes

 TODO: Put seeks inside encode() and decode(), strange
 otherwise. --> Not strange because if you write to
 file then this seek is not needed.

## Further improvements

- [Embarrassingly parallel Huffman encoding](http://www.ittc.ku.edu/~jsv/Papers/HoV95.pdcfull.pdf)

## Resources

-   [Explanation of Huffman Encoding - Stanford course](https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1126/handouts/220%20Huffman%20Encoding.pdf)
-   Idea of using [a static FSM decoding table for HTTP headers](https://github.com/python-hyper/hpack/blob/master/src/hpack/huffman_table.py) ([RFC standard for HPACK](https://www.rfc-editor.org/rfc/rfc7541#appendix-B))
- [Data-Parallel Finite-State Machines by Microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/asplos302-mytkowicz.pdf)
