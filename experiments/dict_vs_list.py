import string


def is_contiguous():
    letters = string.printable

    lst = sorted([ord(l) for l in letters])

    for i in range(len(lst) - 1):
        if lst[i+1] - lst[i] > 1:
            print("TOO BAD")
            print(i)

    print(lst)


def main():
    letters = string.ascii_lowercase

    dct = {l: 10 for l in letters}
    lst = [10 for _ in letters]

    N = 10**7

    # NOTE: 6.12s
    # for _ in range(N):
    #     for l in letters:
    #         dct[l]

    # NOTE: About 2x slower.
    # NOTE: 10.83s
    # ord_ = ord
    # for _ in range(N):
    #     for l in letters:
    #         lst[ord_(l) - 97]

    # NOTE: 1.38s
    # for _ in range(N):
    #     map(lambda x: ord(x) - 97, letters)

    # NOTE: 1.08s
    # for _ in range(N):
    #     map(dct.__getitem__, letters)


if __name__ == "__main__":
    main()

    # NOTE: No problem since we can just create a 256 len list to contain
    # all chars. No need to be contiguous.
    # is_contiguous()
