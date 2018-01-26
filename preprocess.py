from common.Cube import *
from common.config import *


if __name__ == '__main__':
	cube = Cube.load_cube(args.cube_file)
	cube.id_to_author = {}
	for id, cell in cube.id_to_cell.items():
		cube.id_to_author[id] = set()
		v, y = cell
		papers = cube.cell_venue[v] & cube.cell_year[y]
		for paper in papers:
			for author in cube.paper_author[paper]:
				cube.id_to_author[id].add(author.replace(' ', '-'))
	with open(args.cube_file, 'wb') as f:
		pickle.dump(cube, f)
