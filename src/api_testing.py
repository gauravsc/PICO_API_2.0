import requests
import json

# api-endpoint 
# URL = "http://52.178.218.172:5000/api/pico/concepts"
URL = "http://52.178.218.172:5000/api/pico/concepts"

# load the train data file
# data = json.load(open('../data/data_with_cuis.json','r')) 

# abstract to label
# abstract_to_label = " ".join([data[0]['population text'], data[0]['intervention text'], data[0]['outcome text']]) 
abstract_to_label = ''' Adults with mild cognitive impairment as defined by each study. Different definitions of mild cognitive impairment were acceptable but they had to be in line with the generally accepted criteria of a subjective memory complaint and relatively preserved daily functioning (e.g. Petersen 1999). Cholinesterase inhibitors of all types, at all doses, and in any formulation for a minimum of one month. We will specify no maximum duration of treatment. The comparator group is to be placebo. Progression to dementia, either in general or specific subtypes: Alzheimer's disease defined by the criteria from the National Institute of Neurological and Communicative Disorders and Stroke and the Alzheimer's Disease and Related Disorders Association (NINCDS-ARDRA; McKhann 1984); vascular dementia defined by consensus criteria (Roman 1993); or dementia with Lewy bodies defined by consensus criteria (McKeith 2005), measured at the time points of 12, 24 and 36 months. Criteria from the fourth edition of the American Psychiatric Association's Diagnostic and Statistical Manual of Mental Disorders (DSM-IV) or the tenth revision of the World Health Organization's (WHO) International Statistical Classification of Diseases and Related Health Problems (ICD-10) for the dementia syndrome in general or specific subtypes will also be acceptable. Cognition will have been measured with standardised cognitive tests.
Side effects, including gastrointestinal and cardiac. Change in cognitive test scores.
Mortality.'''
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