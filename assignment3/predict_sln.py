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
        args = SaveloadHP.load(arg_file)
        self.device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

        ntokens = len(args.vocab.w2i)
        model = RNNModel(args.model, ntokens, args.emsize, args.nhid,
                         args.nlayers, args.dropout, args.tied, args.bidirect).to(self.device)
        model.load_state_dict(torch.load(model_file))
        model.to(self.device)
        return args, model

    def generate(self, max_len=1000):
        sos_idx = self.args.vocab.w2i[SOS]
        sos_tensor = Data2tensor.idx2tensor([sos_idx], self.device).reshape(1, -1)
        hidden = self.model.init_hidden(sos_tensor.size(0))
        doc = [SOS]
        for i in range(max_len):
            score, hidden = self.model(sos_tensor, hidden)
            pred_prob, pred_idx = self.model.inference(score, k=1)
            pred_wd = self.args.vocab.i2w[pred_idx.item()]
            if pred_wd == EOS:
                break
            doc += [pred_wd]
        doc += [EOS]
        return " ".join(doc)

    def recommend(self, context="", topk=5):
        context_idx = [self.args.vocab.w2i[SOS]] + \
                      [self.args.vocab.w2i.get(tok, self.args.vocab.w2i[UNK]) for tok in context.split()]
        context_tensor = Data2tensor.idx2tensor(context_idx, self.device).reshape(1, -1)
        hidden = self.model.init_hidden(context_tensor.size(0))
        score, hidden = self.model(context_tensor, hidden)
        pred_prob, pred_idx = self.model.inference(score[:, -1, :], k=topk)
        rec_wds = [self.args.vocab.i2w[tok_id.item()] for tok_id in pred_idx.squeeze()]
        rec_probs = pred_prob.squeeze().tolist()
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
