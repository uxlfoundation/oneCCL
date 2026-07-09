FROM python:3

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY build_doc.sh Doxyfile copyright cclEULA.txt /app/

RUN apt-get update && apt-get install -y doxygen

CMD [ "./build_doc.sh" ]