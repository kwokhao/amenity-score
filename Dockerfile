# execute this from the root of the git directory
FROM continuumio:miniconda3  # python3 automatically installed

# 1a. install Julia
RUN apt-get update
RUN apt-get install -y wget
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.3-linux-x86_64.tar.gz
RUN tar -xvzf julia-1.5.3-linux-x86_64.tar.gz
RUN mv julia-1.5.3/ opt/
RUN ln -s /opt/julia-1.5.3/bin/julia /usr/local/bin/julia
RUN rm julia-1.5.3-linux-x86_64.tar.gz  # cleanup

# 1b. install Python packages
RUN pip3 install numpy matplotlib seaborn pandas statsmodels geopandas contextily descartes

# 1c. clone the amenity score git repository
RUN git clone https://github.com/kwokhao/amenity-score.git

# 1d. install gpg to decrypt HDB demographics file (encode: gpg -c demog.csv)
RUN apt-get install -y gpg
COPY password.txt /amenity-score/
RUN cd amenity-score/
RUN PWD=`cat password.txt`
RUN gpg --batch --passphrase ${PWD} -d make_data/demog.csv.gpg > make_data/cleanedHDBDemographics.csv
RUN rm make_data/demog.csv.gpg  # cleanup

# 1d. install Julia packages (do this after Python)
RUN cd code/
RUN julia install.jl

# 2. import functions and run the script directly
RUN julia run.jl

# 3. to copy images and csv: docker cp $container_id:/amenity-score/make_data $destination_path
# find the container id: container_id=`docker ps -aq | head -n 1`


# MISCELLANEOUS SAMPLE CODE (ignore):
# CMD [ "python", "./tele-df_actual.py" ]
