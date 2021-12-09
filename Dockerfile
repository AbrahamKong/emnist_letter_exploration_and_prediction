FROM python:3.8-slim-buster
WORKDIR /app

COPY requirements.txt /
RUN pip3 install -r /requirements.txt
COPY . /app

CMD streamlit run app.py
