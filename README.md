# carPrice

you can run the application using docker container
steps:
1: Clone the repo to your local repository.
2: download car_prices.csv using this link: https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data/download?datasetVersionNumber=1
3: run (run all cells) the priceEstimator file
4: create image
docker build -t car-price-app .
5: run the image in container
docker run -p 8501:8501 car-price-app
6: open URL below:
http://localhost:8501
