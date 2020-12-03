from comppy.clt import *
import json

# Begin by defining your ply orientation and play material layups:
layup_orientations = "[0/45/-45/90]"
layup_materials = "[material1/material2/material1/material1]"
# Generate a layup object.
layup_1 = Layup.gen_from_short(layup_orientations, layup_materials)
# View the contents of the
print(layup_1.layup_list)
material1 = Material('material1', 70, 70, 35, 0.13, 0.1)
material2 = Material('material2', 140, 72, 35, 0.13, 0.15)
material3 = Material('material3', 73, 73, 35, 0.13, 0.2)
library1 = MatLib(material1, material2, material3)
laminate1 = Laminate(layup_1, library1)
print('ABD Matrix:')
print(np.array2string(laminate1.abd, max_line_width=np.inf))
print('D Matrix:')
print(np.array2string(laminate1.abd[3:6, 3:6], max_line_width=np.inf))
print('Effective Properties:')
print(laminate1.effective_props)
req_obj = Requirements(1e-6, 1e-6, 1e-6)

print(laminate1.validate(req_obj))

json_object = json.dumps(laminate1.effective_props, indent = 4)
print(json_object)
