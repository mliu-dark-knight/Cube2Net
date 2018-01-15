import pickle
import re

class DblpCube(object):
	pass

class Cube(object):
	def __init__(self, cube):
		self.cell_venue = {}
		self.cell_year = {}
		self.paper_id = cube.paper_id
		self.paper_author = cube.paper_author

		for venue, papers in cube.cell_venue.items():
			self.cell_venue[re.sub('[, ]', '', venue)] = set(papers)
		for year, papers in cube.cell_year_one.items():
			self.cell_year[year] = set(papers)

		self.cell_to_id = {}
		self.id_to_cell = {}
		id = 0
		for venue, paper1 in self.cell_venue.items():
			for year, paper2 in self.cell_year.items():
				if len(paper1 & paper2) > 0:
					self.cell_to_id[(venue, year)] = id
					self.id_to_cell[id] = (venue, year)
					id +=1

		self.num_cell = len(self.id_to_cell)

		with open('data/cell.txt', 'w') as f:
			for id, cell in self.id_to_cell.items():
				f.write(str(id) + '\t' + cell[0] + ',' + str(cell[1]) + '\n')

if __name__ == '__main__':
	with open('data/step1.pkl', 'rb') as f:
		dblp = pickle.load(f)
	cube = Cube(dblp)
	pickle.dump(cube, open('data/cube.pkl', 'wb'))
