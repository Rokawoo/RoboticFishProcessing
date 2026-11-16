# Fish models

`catfish.3mf` is the model the simulation uses: "Catfish full scale.3mf", the file the stand-in fish were printed from. It is 790 mm long with its barbels, lies on its side, and has its head toward −x (`fish_head: "-x"`, `fish_up: "+z"` in `Simulation/sim.yaml`). The simulation scales it to `fish_length`.

Other models can go here too (set `fish_model`); Git tracks only `catfish.3mf` and this file. Without a model the simulation uses a plain stand-in shape.
