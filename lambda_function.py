#ipython u terminal
#import lambda_function
#lambda_function.predict('https://cdn.nyallergy.com/wp-content/uploads/pineapplenew3.webp')

import tflite_runtime.interpreter as tflite #-for docker image using tflite runtime as shown in course videos
#import tensorflow.lite as tflite
from keras_image_helper import create_preprocessor
 
interpreter = tflite.Interpreter(model_path='fruit-model.tflite')
interpreter.allocate_tensors()
 
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
 
preprocessor = create_preprocessor('xception', target_size=(299, 299))
 
classes = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'kiwi',
 'lemon',
 'lettuce',
 'onion',
 'orange',
 'pear',
 'peas',
 'pineapple',
 'potato',
 'spinach',
 'sweetcorn',
 'tomato',
 'watermelon']

def predict(url):
    X = preprocessor.from_url(url)
 
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
 
    # What happens here is we take an Numpy array and 
    # it will be converted to usual python list with usual python floats.
    float_predictions = preds[0].tolist()
 
    return dict(zip(classes, float_predictions))
 
def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
 
