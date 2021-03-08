import unittest
import networkx as nx
from fix_projections import decompose
from fix_projections import is_component_boundary

class test_decompose(unittest.TestCase):

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
		self.Gg.add_edge('J', 'N')
		self.Gg.add_edge('N', 'O')
		self.Gg.add_edge('O', 'P')

		self.Gg.nodes['A']['breaker'] = True

		self.gp_mappings = {
			'A': [0, 1],
			'B': [2, 3, 4, 5],
			'C': [6, 7, 8],
			'D': [9, 10],
			'E': [11],
			'F': [14, 15, 16, 17],
			'G': [18, 19],
			'H': [20, 21],
			'I': [22],
			'J': [26, 27],
			'K': [28, 29],
			'L': [30, 31, 32],
			'M': [33, 34],
			'N': [35, 36],
			'O': [37, 38],
			'P': [39, 40]
		}
	
	def test_vuurstaek_et_al_case(self):
		self.assertTrue(is_component_boundary(self.Gg, 'A', self.gp_mappings))
		self.assertTrue(is_component_boundary(self.Gg, 'E', self.gp_mappings))
		self.assertTrue(is_component_boundary(self.Gg, 'I', self.gp_mappings))
		self.assertTrue(is_component_boundary(self.Gg, 'G', self.gp_mappings))
		self.assertTrue(is_component_boundary(self.Gg, 'M', self.gp_mappings))
		self.assertTrue(is_component_boundary(self.Gg, 'P', self.gp_mappings))

		self.assertFalse(is_component_boundary(self.Gg, 'B', self.gp_mappings))
		self.assertFalse(is_component_boundary(self.Gg, 'C', self.gp_mappings))
		self.assertFalse(is_component_boundary(self.Gg, 'D', self.gp_mappings))
		self.assertFalse(is_component_boundary(self.Gg, 'F', self.gp_mappings))
		self.assertFalse(is_component_boundary(self.Gg, 'H', self.gp_mappings))
		self.assertFalse(is_component_boundary(self.Gg, 'J', self.gp_mappings))
		self.assertFalse(is_component_boundary(self.Gg, 'K', self.gp_mappings))
		self.assertFalse(is_component_boundary(self.Gg, 'L', self.gp_mappings))
		self.assertFalse(is_component_boundary(self.Gg, 'N', self.gp_mappings))
		self.assertFalse(is_component_boundary(self.Gg, 'O', self.gp_mappings))

		C1 = nx.DiGraph()
		C1.add_edge('A', 'B')
		C1.add_edge('B', 'C')
		C1.add_edge('C', 'D')
		C1.add_edge('D', 'E')
		C1.add_edge('B', 'H')
		C1.add_edge('H', 'I')

		C2 = nx.DiGraph()
		C2.add_edge('E', 'F')
		C2.add_edge('F', 'G')
		C2.add_edge('I', 'J')
		C2.add_edge('J', 'F')
		C2.add_edge('J', 'K')
		C2.add_edge('K', 'L')
		C2.add_edge('L', 'M')
		C2.add_edge('J', 'N')
		C2.add_edge('N', 'O')
		C2.add_edge('O', 'P')

		components = decompose(self.Gg, self.gp_mappings)

		self.assertEqual(2, len(components))
		self.assertEqual(len(components[0].edges), len(C1.edges))

		self.assertEqual(len(components[1].edges), len(C2.edges))

		for edge in components[0].edges:
			self.assertIn(edge, C1.edges)

		for edge in components[1].edges:
			self.assertIn(edge, C2.edges)

if __name__ == '__init__':
	unittest.main()