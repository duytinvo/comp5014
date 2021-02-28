import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, bidirect=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        hd_dim = nhid//2 if bidirect else nhid
        num_directions = 2 if bidirect else 1
        dp = dropout if nlayers >= 2 else 0.0
        
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, hd_dim, nlayers, bidirectional=bidirect, dropout=dp, batch_first=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, hd_dim, nlayers, bidirectional=bidirect, nonlinearity=nonlinearity, dropout=dp, batch_first=True)
        self.decoder = nn.Linear(nhid, ntoken)
        # ignore_index = 0 to ignore calculating loss of PAD label
        self.lossF = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.num_labels = ntoken
        self.rnn_type = rnn_type
        self.nlayers = nlayers
        self.num_directions = num_directions
        self.hd_dim = hd_dim

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers*self.num_directions, bsz, self.hd_dim),
                    weight.new_zeros(self.nlayers*self.num_directions, bsz, self.hd_dim))
        else:
            return weight.new_zeros(self.nlayers*self.num_directions, bsz, self.hd_dim)

    def NLL_loss(self, label_score, label_tensor):
        # batch_loss = self.lossF(label_score, label_tensor)
        batch_loss = self.lossF(label_score.contiguous().view(-1, self.num_labels),
                                label_tensor.contiguous().view(-1, ))
        return batch_loss

    def inference(self, label_score, k=1):
        label_prob = F.softmax(label_score, dim=-1)
        label_prob, label_pred = label_prob.data.topk(k)
        return label_prob, label_pred


if __name__ == '__main__':
    from data_utils import Vocab, Txtfile, Data2tensor, seqPAD, PAD
    cutoff = 5
    wl_th = -1
    batch_size = 16
    bptt = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_files = ["../dataset/train.txt"]
    vocab = Vocab(wl_th=wl_th, cutoff=cutoff)
    vocab.build(data_files, firstline=False)
    word2idx = vocab.wd2idx(vocab.w2i)
    label2idx = vocab.tag2idx(vocab.l2i)

    train_data = Txtfile(data_files[0], firstline=False, source2idx=word2idx, label2idx=label2idx)
    # train_data = [sent[0] for sent in train_data]
    train_batch = vocab.minibatches(train_data, batch_size=batch_size)
    inpdata=[]
    outdata=[]
    for sent in train_batch:
        word_pad_ids, seq_lens = seqPAD.pad_sequences(sent, pad_tok=vocab.w2i[PAD])
        data_tensor = Data2tensor.idx2tensor(word_pad_ids)
        for i in range(0, data_tensor.size(1)-1, bptt):
            data, target = vocab.bptt_batch(data_tensor, i, bptt)
            inpdata.append(data)
            outdata.append(target)
        break

    rnn_type = "GRU"
    ntoken = len(vocab.w2i)
    ninp = 32
    nhid = 64
    nlayers = 1
    dropout = 0.5
    tie_weights = False
    bidirect = False

    model = RNNModel(rnn_type=rnn_type, ntoken=ntoken, ninp=ninp, nhid=nhid, nlayers=nlayers,
                     dropout=dropout, tie_weights=tie_weights, bidirect=bidirect)
    hidden = model.init_hidden(batch_size)
    output, hidden = model(inpdata[10], hidden)
    target = outdata[10]
    f_loss = model.NLL_loss(output, target)
