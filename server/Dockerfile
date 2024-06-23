FROM tensorflow/tensorflow:2.12.0

WORKDIR /app

COPY src/ /app/
COPY model/ /app/model/

RUN pip install fastapi uvicorn pillow
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
