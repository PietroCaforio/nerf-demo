.PHONY: data deps

data:
	@echo "Downloading data..."
	mkdir -p data
	cd data && \
	wget https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip && \
	unzip nerf_example_data.zip

deps:
	@echo "creating environment"
	conda env create -f environment.yml && \
	conda activate nerf-demo 