import typing as t


T = t.TypeVar("T")

def get_combinations(*iterables: list[T]) -> list[T]:
    def helper(i: int) -> None:
        if i == N:
            combinations.append(combination[:])
            return

        for elt in iterables[i]:
            combination.append(elt)
            helper(i+1)
            combination.pop()

    if not iterables:
        return []

    N = len(iterables)
    combinations = []
    combination = []
    helper(0)
    return combinations


def main():
    ans = get_combinations([1, 2, 3], [4, 5], [6], [7, 8])
    print(ans)


if __name__ == "__main__":
    main()
