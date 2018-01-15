from config import *
from Agent import Agent
from Environment import *
from Buffer import *

if __name__ == '__main__':
	environment = Environment(args)
	agent = Agent(args, environment)
	agent.train()
	agent.play()
