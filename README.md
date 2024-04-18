# carPrice

you can run the application using docker container
steps: 
1: create image
  docker build -t car-price-app .
2: run the image in container
  docker run -p 8501:8501 car-price-app
3: open URL below: 
  http://localhost:8501
  
