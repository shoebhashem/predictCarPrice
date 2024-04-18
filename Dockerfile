FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501

# Run carprice.py when the container launches
CMD ["streamlit", "run", "carprice.py"]
