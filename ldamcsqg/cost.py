class Cost:

    @staticmethod
    def theorem_3(base, extra):
        return 4 * (base - 1) ** 2 + 96 * base + 32 * extra - 208 - 64 * (base - 2)

    @staticmethod
    def theorem_1(base, extra):
        return -28 * (base - 1) ** 2 + 2 * (base - 1) * (16 * extra - 40)

    @staticmethod
    def exact_decomposition(n):
        return 4 * n ** 2 - 12 * n + 10

    @staticmethod
    def original_su2(k, p):
        return p * (16 * (k + 1) - 40)

    @staticmethod
    def mt_su2(n, n_target):
        return 16 * (n + 1) - 40 + 16 * (n_target - 1)


