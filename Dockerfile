FROM python:3.12

COPY * /opt/microservices/
COPY requirements.txt /opt/microservices/
RUN pip install --upgrade pip \
  && pip install --upgrade pipenv\
  && apt-get clean \
  && apt-get update \
  && apt install -y build-essential \
  && apt install -y libmariadb3 libmariadb-dev \
  && pip install --upgrade -r /opt/microservices/requirements.txt

CMD ["python", "app.py"]