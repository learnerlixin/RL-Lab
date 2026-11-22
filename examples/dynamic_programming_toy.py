from algorithms.tabular.dynamic_programming import value_iteration


def main():
    transitions = [
        [[(1.0, 0, 1.0, True)], [(1.0, 1, 0.0, False)]],
        [[(1.0, 1, 2.0, True)], [(1.0, 1, 2.0, True)]],
    ]
    values, policy = value_iteration(transitions, 2, 2, gamma=0.9)
    print("V*:", values)
    print("greedy actions:", policy)


if __name__ == "__main__":
    main()
