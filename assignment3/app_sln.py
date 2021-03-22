"""
Created on 2018-12-03
@author: duytinvo
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_sln import LMInference

model_api = LMInference(arg_file="./results/lm.args", model_file="./results/lm.m")

# define the app
app = Flask(__name__)
CORS(app)  # needed for cross-domain requests, allow everything by default


@app.route('/getgenerate', methods=['GET'])
def getgenerate():
    """
    GET request at a sentence level
    usage: http://0.0.0.0:5000/getgenerate?max_len=30
    """
    max_len = request.args.get('max_len', default='100', type=str)
    output_data = model_api.generate(max_len=int(max_len))
    app.logger.info("model_output: " + str(output_data))
    response = jsonify(output_data)
    return response


@app.route('/getrecommend', methods=['GET'])
def getrecommend():
    """
    GET request at a sentence level
    usage: http://0.0.0.0:5000/getrecommend?context=i%20went%20to&topk=6
    """
    context = request.args.get('context', default='', type=str)
    topk = request.args.get('topk', default='5', type=str)
    app.logger.info("api_input: " + context)
    output_data = model_api.recommend(context, topk=int(topk))
    output_data = dict(output_data)
    app.logger.info("model_output: " + str(output_data))
    response = jsonify(output_data)
    return response


@app.route('/')
def index():
    return "Language Model Main Page"


# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    """
    kill -9 $(lsof -i:5000 -t) 2> /dev/null
 
    """

    app.run(host='0.0.0.0', debug=True)
