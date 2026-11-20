# Models

- `catfish.3mf`: the file the stand-in fish were printed from (790 mm with its barbels); the simulation scales it to `fish_length`. Other models can go here (`fish_model`); Git tracks only this one.
- `gripper.stl`: the gripper as built, in the FR5's tip link frame, from its CAD model by `python Simulation/import_gripper.py "<model>.stl"`, which prints the TCP and cup layout measured on it (`tool.tcp`, `cup_rows`, `cup_columns`, `cup_footprint_m` must match; a test checks). After changing it, also run `python Simulation/make_collision_model.py`.
- `fr5/`: FAIRINO's FR5 meshes, downloaded (not in Git: no license).
