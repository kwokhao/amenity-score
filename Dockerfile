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
RUN pip3 install numpy matplotlib seaborn pandas statsmodels

# 1c. clone the amenity score git repository
RUN git clone https://github.com/kwokhao/amenity-score.git
RUN cd amenity-score

# 1c. Install Julia packages (do this after Python)
RUN julia install.jl

# to copy images: docker cp $container_id:$source_path $destination_path


# COPY tele-df_actual.py KiasuAgent-a4f9a50e5c83.json kiasuagent-tejnjc-a04ac97efee3.json /
# COPY tele-df_actual.py SIT-RCD-2020-dialogflow-3314.json /
# RUN pip install telepot
# RUN pip install dialogflow
# RUN pip install google-api-core
# RUN pip install python-google-places
# RUN pip install google-cloud-bigquery
# CMD [ "python", "./tele-df_actual.py" ]
