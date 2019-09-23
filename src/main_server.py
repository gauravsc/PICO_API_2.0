from flask import Flask
from flask import jsonify
from flask import request
import json
from pico_annotator import PICOModel

app = Flask(__name__)

# initialize the pico annotation model
pico_model = PICOModel()

@app.route('/api/pico/concepts', methods=['POST'])
def get_concepts():
	# extract the post request data
	data = request.json

	# get the assigned labels
	result = pico_model.predict(data)
	# return the result
	return json.dumps(result)

if __name__ == '__main__':
    
    app.run(debug=True)

    