FROM python:3.8.3-slim-buster
WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip3 install pydot

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip3 install python-multipart
RUN pip3 install pillow
COPY . /code/application
ENV PYTHONPATH "${PYTHONPATH}:/code/application"

RUN rm -rf ~/.keras/datasets/*

CMD python3 ./application/main.py

EXPOSE 9050
