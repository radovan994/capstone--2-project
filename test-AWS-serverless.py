import requests

url = 'https://2zsj2m0u3c.execute-api.eu-north-1.amazonaws.com/test'
data = {'url' : 'https://cdn.nyallergy.com/wp-content/uploads/pineapplenew3.webp'}

result = requests.post(url, json=data).json()
print(result) 