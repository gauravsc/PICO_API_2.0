import requests
import json

# api-endpoint 
# URL = "http://52.178.218.172:5000/api/pico/concepts"
URL = "http://52.178.218.172:5000/api/pico/concepts"

# load the train data file
data = json.load(open('../data/data_with_cuis.json','r')) 

# abstract to label
abstract_to_label = " ".join([data[0]['population text'], data[0]['intervention text'], data[0]['outcome text']]) 
print (abstract_to_label)

# defining a data dict for the post request to be sent to the API 
data = [{'abstract': abstract_to_label}]

# headers 
headers ={
	'Content-Type' : 'application/json'
}
  
# sending get request and saving the response as response object 
res = requests.post(url=URL, data=json.dumps(data), headers=headers) 

# return results 
print (res.text)