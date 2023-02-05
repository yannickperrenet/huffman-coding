2 passes for encoding:
- freq table
- encoding (actually I do this rather dumb so requires another pass)

1 pass decoder using FSM, decoding `DECODER_WORD_SIZE` bits at once.


## Todo

- [x] There must be a smarter way for the encoding part.
- [ ] Working with streams
- [ ] Working with files that don't fit into memory
- [ ] Benchmark for different `DECODER_WORD_SIZE`
- [ ] Benchmark against other Huffman Python implementations on PyPi
- [x] Some tests
- [ ] Implement some more performance improvements as per the Microsoft paper (see resources)
- [ ] It was mentioned somewhere that huffman encoding/decoding is embarrasingly parallel. Let's add
  a `num_jobs` configuration parameter that allows you to specify the number of processes. Although
  in Python I don't expect it to be faster since the GIL only allows 1 thread to be running at a
  time.

## Resources

-   [Explanation of Huffman Encoding - Stanford course](https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1126/handouts/220%20Huffman%20Encoding.pdf)
-   Idea of using [a static FSM decoding table for HTTP headers](https://github.com/python-hyper/hpack/blob/master/src/hpack/huffman_table.py) ([RFC standard for HPACK](https://www.rfc-editor.org/rfc/rfc7541#appendix-B))
- [Data-Parallel Finite-State Machines by Microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/asplos302-mytkowicz.pdf)
