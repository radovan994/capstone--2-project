## About 
A simple CNN model made to classify different types of fruits. 

## Dataset 
https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition

Context

This dataset encompasses images of various fruits and vegetables, providing a diverse collection for image recognition tasks. The included food items are:

Fruits: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango
Vegetables: Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant
Content

The dataset is organized into three main folders:

Train: Contains 100 images per category.

Test: Contains 10 images per category.

Validation: Contains 10 images per category.

Each of these folders is subdivided into specific folders for each type of fruit and vegetable, containing respective images.


## Explanation

Two models were explored. The Xception model showed better crude results and was selected for parameter tuning and final training. All the parameters were explored, some were corrected some were not
since they did not improve the accuracy. EDA, selection and model training can be reviewed in the EDA-and-model-training.ipynb jupyter notebook. This notebook was converted to Python script under the name Untitled.py

The model was then converted to tflite, process explained in the converting-to-tflite.ipynb jupyter notebook and saved as fruit-model.tflite. The orignal KERAS models were too large to upload so were left out of this repository.

testing-tflite.ipynb and testing.ipynb are not important for this project, they were used for internal testing.


# Execution
The dependency file req.txt is a conda environment that I use for multiple projects, it's advisable to just use your tf environment.


For building docker image and testing it, refer to section below. Test images can be found simply by Googling the desired fruit. 
Likewise, you can also test the deployed AWS Lambda function with the API access link in test-AWS-serverless.py, alternatively you can follow section below to test the API. Below image shows testing of the API within AwS:
![lambda image](https://i.imgur.com/Dw7l7fD.png)

## To test the model locally (without Docker): 
```
Open console, Ipython 
import lambda_function
url ='https://organicmandya.com/cdn/shop/files/Pineapple.jpg'
lambda_function.predict(url)
```

## Test with Docker locally:
1. Build docker image `docker build -t fruit-model .  
2. Run the image `docker run -it --rm -p 8080:8080 fruit-model:latest`
3. Test with following command `python test.py` 

## Test the lambda function with AWS API: 
In console, run `test-AWS-serverless.py`
