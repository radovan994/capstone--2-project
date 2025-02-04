import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
data = {'url' : 'https://cdn.nyallergy.com/wp-content/uploads/pineapplenew3.webp'}

result = requests.post(url, json=data).json()
print(result) 