# Computing Amenity Scores

This repository contains files that allow one to replicate the computation of
amenity scores over time for resale HDB flats in Singapore from Mar 2015 -
Feb 2019. The file structure is:
- `code`: contains all code files
- `data`: contains all data files, except for `cleanedHDBDemographics.csv`
- `make_data`: will contain all output CSVs and plots after the code is run.

There are two ways the code may be run:
1. There exists a companion docker image, named `amenityscoreimage.tar`, which
  is hosted elsewhere for security and file size reasons. This image is compiled
  as of January 24, 2021, and initializes a Julia environment with all
  prerequisite packages installed.
  - Copy the image to disk, then load it using
  `docker image load --input PATH/TO/amenityscoreimage.tar`.
  - Run the image: `docker run -it amenityscoredemo`.
  - Julia will automatically start.
  - (*) Run `include("./amenity-score/code/run.jl")` to import the relevant files.
  Run `main()` to start estimating amenity scores; alternatively, follow the
  options detailed in the file to provide keyword arguments to the file.
  - After running the code, to extract the plots and .csv file with the
  estimated amenity scores, quit to your local shell environment. Identify your
  container ID (e.g. ``` container_id=`docker ps -aq | head -n 1` ```), then
  copy the files into your local environment (e.g. `docker cp
  $container_id:/amenity-score/make_data $destination_path`). The data files can
  be extracted similarly: `docker cp $container_id:/amenity-score/data
  $destination_path`.
2. Alternatively, you may wish to set up your own Julia (and embedded Python)
   environment. It suffices to clone this repository, then
  - run `julia amenity-score/code/install.jl` to get the prerequisite packages;
  - decrypt `demog.csv.gpg` in the `make_data` directory and rename it to
	 `cleanedHDBDemographics.csv`; and finally
  - start `julia`, then run `include("./amenity-score/code/run.jl")` and follow
	the instructions beginning in (*). The outputs will automatically be written
	to your local `make_data` subdirectory.
