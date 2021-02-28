"""
Created on: 2021-02-06
Author: duytinvo
"""
import torch
from utils.core_nns import RNNModel
from utils.data_utils import Txtfile, Data2tensor
from utils.data_utils import SOS, EOS, UNK
from utils.data_utils import SaveloadHP


class LMInference:
    def __init__(self, arg_file="./results/lm.args", model_file="./results/lm.m"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args, self.model = self.load_model(arg_file, model_file)

    def load_model(self, arg_file="./results/lm.args", model_file="./results/lm.m"):
        """
        Inputs:
            arg_file: the argument file (*.args)
            model_file: the pretrained model file
        Outputs:
            args: argument dict
            model: a pytorch model instance
        """
        args, model = None, None
        #######################
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
        #######################
        return args, model

    def generate(self, max_len=1000):
        """
        Inputs:
            max_len: max length of a generated document
        Outputs:
             the text form of a generated document
        """
        doc = [SOS]
        #######################
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
        #######################
        doc += [EOS]
        return " ".join(doc)

    def recommend(self, context="", topk=5):
        """
        Inputs:
            context: the text form of given context
            topk: number of recommended tokens
        Outputs:
            A list form of recommended words and their probabilities
                e,g, [('i', 0.044447630643844604),
                     ('it', 0.027285737916827202),
                     ("don't", 0.026111900806427002),
                     ('will', 0.023868300020694733),
                     ('had', 0.02248169668018818)]
        """
        rec_wds, rec_probs = [], []
        #######################
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
        #######################
        return list(zip(rec_wds, rec_probs))


if __name__ == '__main__':
    arg_file = "./results/lm.args"
    model_file = "./results/lm.m"
    lm_inference = LMInference(arg_file, model_file)

    max_len = 20
    doc = lm_inference.generate(max_len=max_len)
    print("Random doc: {}".format(doc))
    context = "i went to school"
    topk = 5
    rec_toks = lm_inference.recommend(context=context, topk=topk)
    print("Recommended words of {} is:".format(context))
    for wd, prob in rec_toks:
        print("\t- {} (p={})".format(wd, prob))
    pass
