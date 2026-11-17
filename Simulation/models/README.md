# Models

`catfish.3mf`: the file the stand-in fish were printed from ("Catfish full scale.3mf"). 790 mm long with its barbels, lying on its side, head toward −x (`fish_head`, `fish_up` in `Simulation/sim.yaml`); the simulation scales it to `fish_length`. Other fish models can go here (`fish_model`); Git tracks only this one. No model: a plain stand-in shape.

`gripper.stl`: the gripper as built (cups, plate, D405 on its bracket, from the flange down), made from its CAD model by `python Simulation/import_gripper.py "<model>.stl"`: in the FR5's tip link frame in meters (flange face at z = 0.1, cups toward +z, x along the cup rows), cut to 100k triangles. The script prints the TCP, cup layout and footprint measured on it; `tool.tcp`, `cup_rows`/`cup_columns` and `cup_footprint_m` must match (a test checks).

`fr5/`: FAIRINO's FR5 link meshes once downloaded (not in Git: no license).
