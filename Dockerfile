# execute this from the root of the git directory
FROM continuumio:miniconda3  # python3 automatically installed

# install Julia
RUN apt-get update
RUN apt-get install -y wget
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.3-linux-x86_64.tar.gz
RUN tar -xvzf julia-1.5.3-linux-x86_64.tar.gz
RUN mv julia-1.5.3/ opt/
RUN ln -s /opt/julia-1.5.3/bin/julia /usr/local/bin/julia
RUN rm julia-1.5.3-linux-x86_64.tar.gz  # cleanup

# install Python packages
RUN pip3 install numpy matplotlib seaborn pandas statsmodels

# to copy images: docker cp $container_id:$source_path $destination_path

# COPY tele-df_actual.py KiasuAgent-a4f9a50e5c83.json kiasuagent-tejnjc-a04ac97efee3.json /
# COPY tele-df_actual.py SIT-RCD-2020-dialogflow-3314.json /
# RUN pip install telepot
# RUN pip install dialogflow
# RUN pip install google-api-core
# RUN pip install python-google-places
# RUN pip install google-cloud-bigquery
# CMD [ "python", "./tele-df_actual.py" ]
