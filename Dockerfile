FROM python:3.8
COPY ./requirements.txt /app/requirements.txt
RUN pip install python-multipart && pip install -r app/requirements.txt && pip install fastapi uvicorn
RUN apt-get update && apt-get install -y python3-opencv && pip install opencv-python
EXPOSE 80
COPY . /app/
WORKDIR app
CMD ["uvicorn", "Captcha_generator.app:application", "--host","0.0.0.0", "--port", "80"]
