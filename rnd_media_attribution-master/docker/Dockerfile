FROM tensorflow/tensorflow:latest-py3
LABEL maintainer="ricardo.savii@dafiti.com.br"

COPY Pipfile Pipfile.lock ./
RUN pip install --no-cache-dir pipenv==2018.11.26 \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        libev-dev=1:4.22-1 \
        wget=1.19.4-1ubuntu2.2 \
        nginx=1.14.0-0ubuntu1.7 \
    && pipenv install --system --deploy --clear \
    && apt-get remove -y gcc python3-dev libssl-dev \
    && apt-get autoremove -y \
    && pip uninstall pipenv -y \
    && rm -rf /var/lib/apt/lists/*

# Set some environment variables. PYTHONUNBUFFERED keeps Python from
# buffering our standard output stream, which means that logs can be
# delivered to the user quickly. PYTHONDONTWRITEBYTECODE keeps Python
# from writing the .pyc files which are unnecessary in this case. We
# also update PATH so that the train and serve programs are found when
# the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program/:${PATH}"
ENV ENVIRON="Docker"

# Set up the program and config in the image
COPY src /opt/program

WORKDIR /opt/program
RUN chmod +x /opt/program/train
