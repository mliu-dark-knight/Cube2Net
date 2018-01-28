import pickle


class DblpCube(object):
	pass

class Cube(object):
	def __init__(self, cube):
		self.cell_venue = {}
		self.cell_year = {}
		self.cell_topic = {}
		self.paper_author = cube.paper_author
		for idx, authors in enumerate(cube.cell_topic):
			self.cell_topic[idx] = authors
		for idx, authors in enumerate(cube.cell_venue):
			self.cell_venue[idx] = authors
		with open('data/year_name.txt') as f:
			for idx, line in enumerate(f):
				self.cell_year[int(line.rstrip())] = cube.cell_year[idx]

		self.id_to_cell = []
		for topic, author_t in self.cell_topic.items():
			print(topic)
			for venue, author_v in self.cell_venue.items():
				for year, author_y in self.cell_year.items():
					author_c = set.intersection(author_t, author_v, author_y)
					if len(author_c) >= 100:
						cell = (topic, venue, year)
						self.id_to_cell.append(cell)

		self.id_to_author = [None for i in self.id_to_cell]

if __name__ == '__main__':
	with open('data/step3.pkl', 'rb') as f:
		dblp = pickle.load(f)
	cube = Cube(dblp)
	print(len(cube.id_to_cell))
	pickle.dump(cube, open('data/cube.pkl', 'wb'))
