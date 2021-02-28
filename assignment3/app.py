"""
Created on 2018-12-03
@author: duytinvo
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import LMInference

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
    doc = ""
    #######################
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE
    #######################
    app.logger.info("model_output: " + str(doc))
    response = jsonify(doc)
    return response


@app.route('/getrecommend', methods=['GET'])
def getrecommend():
    """
    GET request at a sentence level
    usage: http://0.0.0.0:5000/getrecommend?context=i%20went%20to&topk=6
    """
    rec_toks = None
    #######################
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE
    #######################
    rec_toks = dict(rec_toks)
    app.logger.info("model_output: " + str(rec_toks))
    response = jsonify(rec_toks)
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
