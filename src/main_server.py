from flask import Flask
from flask import jsonify
from flask import request
from pico_annotator import PICOModel
app = Flask(__name__)

@app.route('/api/pico/concepts', methods=['POST'])
def get_concepts():
	# extract the post request data
	data = request.get_json()['data']
	# get the assigned labels
	result = pico_annotator.predict(data)
	# return the result
	return jsonify(result)

if __name__ == '__main__':
    
    app.run(debug=True)

    # initialize the pico annotation model
    pico_model = PICOModel()