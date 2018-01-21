from common.Environment import *


class Environment(Base):
	def __init__(self, params):
		super(Environment, self).__init__(params)

	def trajectory_reward(self, state, actions):
		state = set([self.id_to_cell[id] for id in state])
		if self.cube is None:
			self.cube = Cube.load_cube(self.params.cube_file)
		return self.cube.trajectory_reward(state, [self.id_to_cell[id] for id in actions])
