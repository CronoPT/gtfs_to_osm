import unittest
from utils.geometric_utils import lines_intersect

class test_lines_intersect(unittest.TestCase):

	def setUp(self):
		self.line_a = [[2,  1], [5,  1]]
		self.line_b = [[3, -1], [3,  3]]
		self.line_c = [[2,  2], [5,  2]]
		self.line_d = [[2,  4], [4, -1]]
		self.line_e = [[4, -1], [2,  4]]

		self.line_f = [[-4, 5], [-1, 5]]
		self.line_g = [[-1, 5], [-1, 1]]
		self.line_h = [[-1, 1], [-4, 1]]
		self.line_i = [[-4, 1], [-4, 5]]
		self.line_j = [[-3, 3], [-5, 3]]

	def test_lines_intersect(self):
		self.assertTrue(lines_intersect(self.line_a, self.line_b))
		self.assertTrue(lines_intersect(self.line_b, self.line_c))
		self.assertTrue(lines_intersect(self.line_d, self.line_a))
		self.assertTrue(lines_intersect(self.line_d, self.line_b))
		self.assertTrue(lines_intersect(self.line_d, self.line_c))
		self.assertTrue(lines_intersect(self.line_e, self.line_b))
		self.assertTrue(lines_intersect(self.line_e, self.line_c))
		self.assertTrue(lines_intersect(self.line_j, self.line_i))


		self.assertFalse(lines_intersect(self.line_a, self.line_c))
		self.assertFalse(lines_intersect(self.line_j, self.line_f))
		self.assertFalse(lines_intersect(self.line_j, self.line_g))
		self.assertFalse(lines_intersect(self.line_j, self.line_h))

if __name__ == '__init__':
	unittest.main()