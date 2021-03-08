import unittest
import networkx as nx
from fix_projections import find_non_bifurcating
from fix_projections import is_boundary

class test_find_non_bifurcating(unittest.TestCase):

	def setUp(self):
		self.Gg = nx.DiGraph()
		self.Gg.add_edge('A', 'B')
		self.Gg.add_edge('B', 'C')
		self.Gg.add_edge('C', 'D')
		self.Gg.add_edge('D', 'E')
		self.Gg.add_edge('E', 'F')
		self.Gg.add_edge('F', 'G')
		self.Gg.add_edge('B', 'H')
		self.Gg.add_edge('H', 'I')
		self.Gg.add_edge('I', 'J')
		self.Gg.add_edge('J', 'K')
		self.Gg.add_edge('K', 'L')
		self.Gg.add_edge('L', 'M')
		self.Gg.add_edge('J', 'F')

		self.Gg.nodes['A']['breaker'] = True

		self.gp_mappings = {
			'A': [0, 1],
			'B': [2, 3, 4, 5],
			'C': [6, 7, 8],
			'D': [9, 10],
			'E': [11, 12, 13],
			'F': [14, 15, 16, 17],
			'G': [18, 19],
			'H': [20, 21],
			'I': [22, 23, 24, 25],
			'J': [26, 27],
			'K': [28, 29],
			'L': [30, 31, 32],
			'M': [33, 34],
		}

	def test_vuurstaek_et_al_case(self):

		self.assertTrue(is_boundary(self.Gg, 'A', self.gp_mappings))
		self.assertTrue(is_boundary(self.Gg, 'B', self.gp_mappings))
		self.assertTrue(is_boundary(self.Gg, 'F', self.gp_mappings))
		self.assertTrue(is_boundary(self.Gg, 'G', self.gp_mappings))
		self.assertTrue(is_boundary(self.Gg, 'J', self.gp_mappings))
		self.assertTrue(is_boundary(self.Gg, 'M', self.gp_mappings))
		
		self.assertFalse(is_boundary(self.Gg, 'C', self.gp_mappings))
		self.assertFalse(is_boundary(self.Gg, 'D', self.gp_mappings))
		self.assertFalse(is_boundary(self.Gg, 'E', self.gp_mappings))
		self.assertFalse(is_boundary(self.Gg, 'H', self.gp_mappings))
		self.assertFalse(is_boundary(self.Gg, 'I', self.gp_mappings))
		self.assertFalse(is_boundary(self.Gg, 'K', self.gp_mappings))
		self.assertFalse(is_boundary(self.Gg, 'L', self.gp_mappings))

		right_sequences = [
			['A', 'B'],
			['B', 'C', 'D', 'E', 'F'],
			['F', 'G'],
			['B', 'H', 'I', 'J'],
			['J', 'K', 'L', 'M']
		]

		sequences = find_non_bifurcating(self.Gg, self.gp_mappings)

		self.assertEqual(len(right_sequences), len(sequences))

		for sequence in sequences:
			self.assertIn(sequence, right_sequences)


if __name__ == '__init__':
	unittest.main()