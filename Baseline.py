import numpy as np
from copy import deepcopy
from common.Cube import Cube
from common.config import *


class Baseline(object):
	def __init__(self, params):
		self.params = params
		self.cube = Cube.load_cube(args.cube_file)

	def initial_state(self):
		return set(list(np.random.choice(len(self.cube.id_to_cell), self.params.initial_state_size, replace=False)))

	def random_baseline(self, states):
		rewards = []
		for state in states:
			actions = set(list(np.random.choice(len(self.cube.id_to_cell), self.params.trajectory_length, replace=False)))
			final = state | actions
			rewards.append(self.cube.total_reward([self.cube.id_to_cell[id] for id in final], self.params.measure))
		return np.average(np.array(rewards))

	def greedy_baseline(self, states):
		rewards = []
		for state in states:
			next = deepcopy(state)
			for _ in range(self.params.trajectory_length):
				nexts = []
				for a in self.cube.id_to_cell:
					if a not in next:
						next.add(a)
						nexts.append((next, self.cube.total_reward([self.cube.id_to_cell[id] for id in next], self.params.measure)))
						next.remove(a)
				next = max(nexts, key=lambda e: e[1])[0]
			rewards.append(self.cube.total_reward([self.cube.id_to_cell[id] for id in next], self.params.measure))
		return np.average(np.array(rewards))


if __name__ == '__main__':
	baseline = Baseline(args)
	states = [baseline.initial_state() for _ in range(args.batch_size)]
	print('random baseline: %f' % baseline.random_baseline(states))
	print('greedy baseline: %f' % baseline.greedy_baseline(states))
