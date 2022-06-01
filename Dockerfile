
FROM continuumio/miniconda3

ADD  dist/housing_price_anubhav-0.3.tar.gz home/

WORKDIR /home/housing_price_anubhav-0.3


ADD ./tests ./tests
# ADD ./deploy/conda/linux_cpu_py39.yml ./deploy/conda/linux_cpu_py39.yml
ADD ./logs ./logs
ADD ./docs ./docs
ADD entrypoint.sh .

# RUN conda env create -f deploy/conda/linux_cpu_py39.yml

# RUN conda run -n testing-env pip install .

RUN pip install .
RUN pip install numpy
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install pytest

RUN chmod +x entrypoint.sh

ENTRYPOINT [ "./entrypoint.sh" ]
