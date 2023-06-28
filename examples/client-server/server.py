"""Example HTTP server that decodes a compressed stream of bytes.

Run the server using:

```sh
python examples/client-server/server.py
```

"""

from contextlib import contextmanager
import io
import socket
from concurrent.futures import ThreadPoolExecutor

import huffman_coding

HOST = "localhost"
PORT = 5007
NUM_CLIENTS = 10


@contextmanager
def makefile(s: socket.socket):
    file = s.makefile(mode="rb", newline="", buffering=0)
    try:
        yield file
    finally:
        file.close()


def handle_conn(conn: socket.socket, addr) -> None:
    decoded_text = io.StringIO()

    with conn:
        print("Connected by", addr)

        with makefile(conn) as f_in:
            huffman_coding.decode(f_in=f_in, f_out=decoded_text)

        decoded_text.seek(0)
        print(decoded_text.read())

    print("Done handling", addr)


def main() -> None:
    # Listen for incoming connections.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        # Serve requests indefinitely, each within its own thread.
        with ThreadPoolExecutor(max_workers=NUM_CLIENTS) as executor:
            while True:
                conn, addr = s.accept()
                executor.submit(handle_conn, conn, addr)


if __name__ == "__main__":
    main()
