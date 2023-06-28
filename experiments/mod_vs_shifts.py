if __name__ == "__main__":
    N = 10**8
    x = 3928239**4


    # Pretty much the same
    # for _ in range(N):
    #     y = x - x % 8

    # for _ in range(N):
    #     y = (x >> 3) << 3

    # for _ in range(N):
    #     y = x & -8

    for _ in range(N):
        y = x - (x & 7)
