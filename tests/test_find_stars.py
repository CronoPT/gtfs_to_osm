import unittest
import networkx as nx
from fix_projections import find_stars

class test_find_stars(unittest.TestCase):

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

		right_stars = [
			{
				'center': 'B',
				'before': ['A'],
				'after' : ['C', 'H']
			},
			{
				'center': 'F',
				'before': ['E', 'J'],
				'after' : ['G']
			},
			{
				'center': 'J',
				'before': ['I'],
				'after' : ['F', 'K']
			}
		]

		stars = find_stars(self.Gg, self.gp_mappings)

		# just ensuring that all the list are in the same order
		# so it is easier to compare everything
		for star in right_stars:
			star['before'].sort()
			star['after'].sort()
		for star in stars:
			star['before'].sort()
			star['after'].sort()

		self.assertEqual(len(right_stars), len(stars))

		for star in stars:
			self.assertIn(star, right_stars)


if __name__ == '__init__':
	unittest.main()