import itertools
import pickle
import networkx as nx
import numpy as np


class Cube(object):
	def initial_state(self, path, threshold, debug=False):
		if debug:
			return set(list(np.random.choice(len(self.id_to_cell), 10, replace=False)))
		authors = set()
		with open(path) as f:
			for line in f:
				authors.add(line.rstrip().split('\t')[0].replace('_', ' '))
		ids = []
		for id, _ in enumerate(self.id_to_cell):
			if len(self.cell_authors(id) & authors) > threshold:
				ids.append(id)
		return set(ids)

	# compute reward to go
	def trajectory_reward(self, state, actions, func):
		G = nx.Graph()
		for cell in state:
			self.add_cell(G, cell)
		rewards = [getattr(nx, func)(G)]

		for cell in actions:
			self.add_cell(G, cell)
			rewards.append(getattr(nx, func)(G))
		total = rewards[-1]
		rewards = [total - r for r in rewards]
		return rewards[:-1]

	def cell_authors(self, id):
		if self.id_to_author[id] is None:
			t, v, y = self.id_to_cell[id]
			authors = set.intersection(self.cell_topic[t], self.cell_venue[v], self.cell_year[y])
			self.id_to_author[id] = authors
			return authors
		return self.id_to_author[id]

	# mutate G
	def add_cell(self, G, cell):
		author_c = self.cell_authors(cell)
		for author_p in self.paper_author:
			authors = author_p & author_c
			if bool(authors):
				for pair in itertools.combinations(authors, 2):
					G.add_edge(pair[0], pair[1])

	def total_reward(self, state, func):
		G = nx.Graph()
		for cell in state:
			self.add_cell(G, cell)
		return getattr(nx, func)(G)


	@staticmethod
	def load_cube(path):
		with open(path, 'rb') as f:
			return pickle.load(f)
