import json
import os

def read_json_file(filename, key):
	with open(os.path.join(os.getcwd(), filename), 'r') as stream:
		if key is None:
			return (json.load(stream)).key
		else:
			return json.load(stream)

