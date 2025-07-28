# nerf-demo

This is a minimal NeRF training demo.

Dataset of choice is from [this source (UCSD, ECCV 2020)](http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/)

Training tiny naive NeRF

![Training animation](./output.gif)

Training naive TensoRF with very low rank and resolution

![Training animation](./output_tensorf.gif)
### Requirements
This project uses **Conda** for environment management.

To set up everything:

```bash
make deps      # Create the Conda environment
make data      # Download the dataset
```

### Run Experiments

To run the NeRF baseline (simple MLP):

```bash
python run_nerf.py
```

To run the TensoRF version (tensor decomposition):

```bash
python run_tensorf.py
```
