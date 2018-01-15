import numpy as np


class Environment(object):
	def __init__(self, params):
		self.params = params
		self.load_cell()
		self.load_embed()
		self.load_replay_buffer()

	def load_cell(self):
		self.cell_to_id = {}
		self.id_to_cell = {}
		with open(self.params.cell_file) as f:
			for line in f:
				splits = line.rstrip().split('\t')
				id, venue, year = int(splits[0]), splits[1].split(',')[0], int(splits[1].split(',')[1])
				self.cell_to_id[(venue, year)] = id
				self.id_to_cell[id] = (venue, year)

	def load_embed(self):
		self.v_embed, self.y_embed = {}, {}
		with open(self.params.venue_file) as f:
			for line in f:
				line = line.rstrip().split('\t')
				self.v_embed[line[0]] = np.array(map(float, line[1].split()))
		with open(self.params.year_file) as f:
			for line in f:
				line = line.rstrip().split('\t')
				self.y_embed[int(line[0])] = np.array(map(float, line[1].split()))

	def load_replay_buffer(self):
		self.replay_buffer = []
		with open(self.params.replay_buffer_file) as f:
			for line in f:
				splits = line.rstrip().split('\t')
				state = map(lambda cell: (cell.split()[0], int(cell.split()[1])), splits[0].split())
				action = (splits[1].split()[0], int(splits[1].split()[1]))
				next = map(lambda cell: (cell.split()[0], int(cell.split()[1])), splits[2].split())
				reward = float(splits[3])
				self.replay_buffer.append((state, action, next, reward))

	def sample(self):
		return self.replay_buffer[np.random.choice(len(self.replay_buffer), self.params.batch_size)]

	# state is a set of cell ids
	def next_states(self, state):
		actions = set(self.id_to_cell.keys()) - state
		states = []
		for action in actions:
			states.append(state | action)
		return states

	def state_embed(self, state):
		embed = []
		for cell in state:
			v, y = cell
			embed.append(np.concatenate((self.v_embed[v], self.y_embed[y])))
		return np.mean(np.array(embed))
