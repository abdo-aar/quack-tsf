# Use a base image from NVIDIA that includes CUDA, with Ubuntu
# To be configured following the right cuda version on the host machine
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -t -i -p -y && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt or environment.yml
COPY environment.yml .
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda env create -f environment.yml && \
    echo "conda activate quack-tsf" >> ~/.bashrc && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate quack-tsf && \
    conda install -c conda-forge nano --yes

# Set PYTHONPATH environment variable
ENV PYTHONPATH="/usr/src/app:${PYTHONPATH}"

# Use a custom script or direct command to activate conda environment on start
ENTRYPOINT [ "/bin/bash", "-c", "--" ]
CMD [ "echo Activating quack-tsf && . /opt/conda/etc/profile.d/conda.sh && conda activate quack-tsf && /bin/bash" ]
