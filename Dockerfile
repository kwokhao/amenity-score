# execute this from the root of the git directory
FROM debian:latest

# 1a. install Julia
RUN apt-get update
RUN apt-get install -y wget bzip2
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.3-linux-x86_64.tar.gz
RUN tar -xvzf julia-1.5.3-linux-x86_64.tar.gz
RUN mv julia-1.5.3/ opt/
RUN ln -s /opt/julia-1.5.3/bin/julia /usr/local/bin/julia
RUN rm julia-1.5.3-linux-x86_64.tar.gz  # cleanup

# 1b. install Python and packages (https://github.com/airbnb/knowledge-repo/issues/252)
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p ./miniconda
RUN miniconda/bin/conda install pip
RUN miniconda/bin/pip install numpy matplotlib seaborn pandas statsmodels geopandas contextily descartes

# 1c. clone the amenity score git repository
RUN apt-get install -y git
RUN git clone https://github.com/kwokhao/amenity-score.git

# 1d. install gpg to decrypt HDB demographics file (encode: gpg -c demog.csv)
# RUN apt-get install -y gpg
# COPY password.txt /
# RUN PWD=`cat password.txt`
# RUN gpg --passphrase "${PWD}" -do amenity-score/make_data/cleanedHDBDemographics.csv amenity-score/make_data/demog.csv.gpg
# RUN rm make_data/demog.csv.gpg  # cleanup

# 1d'. 1d not working for now, so copy cleanedHDBDemographics.csv directly
COPY make_data/cleanedHDBDemographics.csv amenity-score/make_data/

# 1d. install Julia packages (do this after Python)
RUN julia ./amenity-score/code/install.jl

# 2. import functions and run the script directly
RUN julia ./amenity-score/code/run.jl

# 3. first function upon load
CMD [ "julia", "./amenity-score/code/run.jl" ]

# 4. to copy images and csv: docker cp $container_id:/amenity-score/make_data $destination_path
# find the container id: container_id=`docker ps -aq | head -n 1`
