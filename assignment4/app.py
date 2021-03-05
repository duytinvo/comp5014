"""
Created on 2018-12-03
@author: duytinvo
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import SentiInference

model_api = SentiInference(arg_file="./results/senti_cls_unilstm.args", model_file="./results/senti_cls_unilstm.m")

# define the app
app = Flask(__name__)
CORS(app)  # needed for cross-domain requests, allow everything by default


@app.route('/getpredict', methods=['GET'])
def getpredict():
    """
    GET request at a sentence level
    usage: http://0.0.0.0:5000/getpredict?doc=i%20love%20it&topk=5
    """
    doc = request.args.get('doc', default='', type=str)
    topk = request.args.get('topk', default='5', type=str)
    app.logger.info("api_input: " + doc)
    output_data = model_api.predict(doc, topk=int(topk))
    output_data = dict(output_data)
    app.logger.info("model_output: " + str(output_data))
    response = jsonify(output_data)
    return response


@app.route('/')
def index():
    return "Sentiment Model Main Page"


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
