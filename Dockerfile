FROM python:3.6.5

# working directory
WORKDIR /usr/src/app

# copy requirement file to working directory
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "./run.py"]


