from multiprocessing import *
from common.Environment import *


class Environment(Base):
	def __init__(self, params):
		super(Environment, self).__init__(params)
		self.sigma = np.std(self.cell_embed, axis=0)

	def reward_multiprocessing(self, state_embeds, initial_states, actions):
		def worker(worker_id):
			for idx, state_embed, initial_state, action in zip(range(len(state_embeds)), state_embeds, initial_states, actions):
				if idx % num_process == worker_id:
					queue.put((state_embed, action, np.array(self.trajectory_reward(initial_state, action))))

		assert len(state_embeds) == len(initial_states) and len(initial_states) == len(actions)
		num_process = 4
		queue = Queue()
		processes = []
		for id in range(num_process):
			process = Process(target=worker, args=(id,))
			process.start()
			processes.append(process)
		for process in processes:
			process.join()
		ret_states, ret_actions, ret_rewards = [], [], []
		while not queue.empty():
			state, action, reward = queue.get()
			ret_states.append(state)
			ret_actions.append(action)
			ret_rewards.append(reward)

		return np.concatenate(ret_states, axis=0), np.concatenate(ret_actions, axis=0), np.concatenate(ret_rewards, axis=0)


	def trajectory_reward(self, state, actions):
		state = set([self.id_to_cell[id] for id in state])
		return self.cube.trajectory_reward(state, [self.id_to_cell[id] for id in actions], self.params.measure)
