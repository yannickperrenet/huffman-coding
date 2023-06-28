# Example: client-server

The client takes `hamlet.txt` from the root directory, encodes it and sends it over a socket to the
server, which will decode it (without using the original `hamlet.txt`) and print it back to the
terminal.

The goal is to illustrate a potential use case where applications first compress the byte stream
that they want to send over a network. This is very common as network communication is often a
bottleneck.

## Running the example
```sh
# Get this example
git clone https://github.com/yannickperrenet/huffman-coding.git && cd huffman-coding

# Be sure to obtain the hamlet.txt example text.
wget https://gist.githubusercontent.com/provpup/2fc41686eab7400b796b/raw/b575bd01a58494dfddc1d6429ef0167e709abf9b/hamlet.txt -O hamlet.txt

# Run the server
python examples/client-server/server.py
# Run the client (in another terminal session)
python examples/client-server/client.py
```
