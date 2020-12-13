from unittest import TestCase
import unittest
import numpy as np
import nose
import IdentifyingSpecialMatrices as ism


class TestIdentifyingSpecialMatrices(TestCase):

    def test_is_singular_true(self):
        A = np.array([
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 4, 4],
            [0, 0, 5, 5]
        ], dtype=np.float_)

        print(f"A = \n{A}")

        result = ism.isSingular(A)
        expect = True

        assert (result == expect)

    def test_is_singular_false(self):
        A = np.array([
            [0, 7, -5, 3],
            [2, 8, 0, 4],
            [3, 12, 0, 5],
            [1, 3, 1, 3]
        ], dtype=np.float_)

        print(f"A = \n{A}")

        result = ism.isSingular(A)
        expect = False

        assert (result == expect)

    def test_DET_is_singular_false(self):
        A = np.array([
            [  0,   7,  -5,   3],
            [  2,   8,   0,   4],
            [  3,  12,   0,   5],
            [  1,   3,   1,   3]
        ], dtype=np.float_)

        print(f"A = \n{A}")

        #det_a = np.linalg.det(A)
        print(f"Determinate of B: {np.linalg.det(A)}")

        # result = ism.isSingular(A)
        # expect = False
        #
        # assert (result == expect)

    def test_fix_row_0(self):
        A = np.array([
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 4, 4],
            [0, 0, 5, 5]
        ], dtype=np.float_)

        expect_matrix = np.array([
            [1, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 4, 4],
            [0, 0, 5, 5]
        ], dtype=np.float_)

        print(f"A = \n{A}")
        print(f"expect_matrix =\n{expect_matrix}")

        result = ism.fixRowZero(A)
        print(f"result = \n{result}")

        assert (np.array_equal(result, expect_matrix))

    def test_fix_row_1(self):
        A = np.array([
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 4, 4],
            [0, 0, 5, 5]
        ], dtype=np.float_)

        expect_matrix = np.array([
            [2, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 4, 4],
            [0, 0, 5, 5]
        ], dtype=np.float_)

        print(f"A = \n{A}")
        print(f"expect_matrix =\n{expect_matrix}")

        result = ism.fixRowOne(A)
        print(f"result = \n{result}")

        assert (np.array_equal(result, expect_matrix))

    def test_fix_row_2(self):
        A = np.array([
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 4, 4],
            [0, 0, 5, 5]
        ], dtype=np.float_)

        expect_matrix = np.array([
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 5, 5]
        ], dtype=np.float_)

        print(f"A = \n{A}")
        print(f"expect_matrix =\n{expect_matrix}")

        result = ism.fixRowTwo(A)
        print(f"result = \n{result}")

        assert (np.array_equal(result, expect_matrix))

    def test_fix_row_3(self):

        A = np.array([
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 4, 4],
            [0, 0, 5, 5]
        ], dtype=np.float_)

        expect_matrix = np.array([
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 4, 4],
            [0, 0, 0, 1]
        ], dtype=np.float_)

        print(f"A = \n{A}")
        # print(f"expect_matrix =\n{expect_matrix}")
        #
        # result = ism.fixRowThree(A)
        # print(f"result = \n{result}")

        self.assertRaises(ism.MatrixIsSingular, ism.fixRowThree(A))
        #assert (np.array_equal(result, expect_matrix))

    # Invertible Matrices: Not Singular
    @unittest.skip
    def test_fix_row_0_NotSingular(self):
        A = np.array([
            [  0,   7,  -5,   3],
            [  2,   8,   0,   4],
            [  3,  12,   0,   5],
            [  1,   3,   1,   3]
        ], dtype=np.float_)

        expect_matrix = np.array([
            [  1,   7.5, -2.5,  3.5 ],
            [  2,     8,    0,    4 ],
            [  3,    12,    0,    5 ],
            [  1,     3,    1,    3 ]
        ], dtype=np.float_)

        print(f"A = \n{A}")
        print(f"expect_matrix =\n{expect_matrix}")

        result = ism.fixRowZero(A)
        print(f"result = \n{result}")

        assert (np.array_equal(result, expect_matrix))

    @unittest.skip
    def test_fix_row_NotSingular(self):
        A = np.array([
            [  1,   7.5, -2.5,  3.5 ],
            [  2,     8,    0,    4 ],
            [  3,    12,    0,    5 ],
            [  1,     3,    1,    3 ]
        ], dtype=np.float_)

        expect_matrix = np.array([
            [  1,   7.5, -2.5,  3.5 ],
            [  0,    -8,    5,   -3 ],
            [  3,    12,    0,    5 ],
            [  1,     3,    1,    3 ]
        ], dtype=np.float_)


        print(f"A = \n{A}")
        print(f"expect_matrix =\n{expect_matrix}")

        result = ism.fixRowOne(A)
        print(f"result = \n{result}")

        assert (np.array_equal(result, expect_matrix))
