"""Example client that compresses a file and sends it over the wire.

Run the client using (be sure to run the server first!):

```sh
python examples/client-server/server.py
# In another terminal
python examples/client-server/client.py
```

"""

from contextlib import contextmanager
import socket

import huffman_coding

HOST = "127.0.0.1"  # localhost
PORT = 5007

@contextmanager
def makefile(s: socket.socket):
    file = s.makefile(mode="wb", newline="", buffering=0)
    try:
        yield file
    finally:
        file.close()


def send_hamlet(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        with (
            open("hamlet.txt", mode="r", newline="") as f_in,
            makefile(s) as f_out,
        ):
            huffman_coding.encode(f_in=f_in, f_out=f_out)

    print("Hamlet was successfully encoded and sent to the server.")


def main():
    send_hamlet(HOST, PORT)


if __name__ == "__main__":
    main()
