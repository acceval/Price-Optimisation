FROM python:3

WORKDIR /app

COPY . /app

# RUN pip install -r requirements.txt
#
# RUN python app.py
#
# CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]


RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

EXPOSE 5050

ENTRYPOINT [ "python", "app.py" ]