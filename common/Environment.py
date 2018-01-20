import numpy as np
from common.Cube import Cube


class Base(object):
	def __init__(self, params):
		self.params = params
		self.cube = None
		self.load_cell()
		self.load_embed()

	def load_cell(self):
		self.cell_to_id = {}
		self.id_to_cell = {}
		with open(self.params.cell_file) as f:
			for line in f:
				splits = line.rstrip().split('\t')
				id, venue, year = int(splits[0]), splits[1].split('|')[0], int(splits[1].split('|')[1])
				self.cell_to_id[(venue, year)] = id
				self.id_to_cell[id] = (venue, year)
		self.action_size = len(self.id_to_cell)

	def load_embed(self):
		v_embed = {}, {}
		with open(self.params.venue_file) as f:
			for line in f:
				line = line.rstrip().split('\t')
				v_embed[line[0]] = np.array(map(float, line[1].split()))
		cell_embed = []
		for id in range(len(self.id_to_cell)):
			cell = self.id_to_cell[id]
			if self.terminal_action(cell):
				cell_embed.append(np.zeros(2 * self.params.embed_dim))
			else:
				cell_embed.append(np.concatenate((v_embed[cell[0]], y_embed[cell[1]])))
		self.cell_embed = np.array(cell_embed)

	def state_embed(self, state):
		return np.mean(self.cell_embed[state], axis=0)

	def terminal_state(self, state):
		return len(state) >= self.params.max_state_size

	def total_reward(self, state):
		if self.cube is None:
			self.cube = Cube.load_cube(self.params.cube_file)
		return self.cube.total_reward([self.id_to_cell[id] for id in state])
