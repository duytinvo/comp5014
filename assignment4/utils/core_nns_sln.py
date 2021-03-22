import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class RNNModule(nn.Module):
    """
    A Recurrent Module could be use to implement [Bi-]RNN, [Bi-]GRU and [Bi-]LSTM
    """
    def __init__(self, rec_type, ntokens, emb_size, hidden_size, nlayers, dropout=0.5, bidirect=False):
        super(RNNModule, self).__init__()
        # Create a dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Create an embedding layer
        self.embedding_layer = nn.Embedding(ntokens, emb_size)

        # Create a recurrent layer
        rec_size = hidden_size//2 if bidirect else hidden_size
        rec_dropout = dropout if nlayers >= 2 else 0.0
        if rec_type in ['LSTM', 'GRU']:
            self.recurrent_layer = getattr(nn, rec_type)(emb_size, rec_size, nlayers, bidirectional=bidirect,
                                                         dropout=rec_dropout, batch_first=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rec_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.recurrent_layer = nn.RNN(emb_size, rec_size, nlayers, bidirectional=bidirect, nonlinearity=nonlinearity,
                                          dropout=rec_dropout, batch_first=True)

        # Randomly initialize embeddings
        self.init_embedding()

    def init_embedding(self, initrange=0.1):
        """
        Initialize word embedding  randomly ranging between [-initrange, initrange]
        """
        self.embedding_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_tensor, input_lengths=None, init_hidden=None):
        # sort lengths of input tensors in the descending mode
        input_tensor, input_lengths, order_tensor, reorder_tensor = self.sort_tensors(input_tensor, input_lengths)
        # Project word embedding vectors [e1,...,en] from index input_tensor [id_w1,..., id_wn]
        emb = self.embedding_layer(input_tensor)
        # Dropout on embedding
        emb_drop = self.dropout_layer(emb)
        # Pass embedding vectors [e1,...,en]  into a recurrent model to learn hidden states [h1, ..., hn]
        rec_output, rec_hidden = self.get_all_hidden(emb_drop, input_lengths, init_hidden)
        # recover the original order of outputs to compute loss
        rec_output = self.resort_tensors(rec_output, reorder_tensor, dim=0)
        rec_hidden = self.resort_tensors(rec_hidden, reorder_tensor, dim=1)
        return rec_output, rec_hidden

    def get_all_hidden(self, emb_inputs, input_lengths=None, init_hidden=None):
        """
        Pass embedding vectors [e1,...,en]  into a recurrent model to learn hidden states [h1, ..., hn]
        input:
            emb_inputs: tensor(batch_size, max_seq_length, emb_size)
            input_lengths: tensor(batch_size,  1)
        output:
            tensor(batch_size, max_seq_length, hidden_dim)
        """
        if input_lengths is not None:
            total_length = emb_inputs.size(1)
            pack_input = pack_padded_sequence(emb_inputs, input_lengths.cpu(), True)
            self.recurrent_layer.flatten_parameters()
            # rec_output = tensor(batch_size, max_seq_length, rnn_dim * num_directions)
            # rec_hidden = h_n or (h_n,c_n); h_n = tensor(num_layers * num_directions, batch_size, rnn_dim)
            rec_output, rec_hidden = self.recurrent_layer(pack_input, init_hidden)
            rec_output, _ = pad_packed_sequence(rec_output, batch_first=True, total_length=total_length)
        else:
            self.recurrent_layer.flatten_parameters()
            rec_output, rec_hidden = self.recurrent_layer(emb_inputs, init_hidden)
        return rec_output, rec_hidden

    @staticmethod
    def get_last_hidden(rec_hidden, bidirect=False):
        """
        Extract the last hidden vector as input representations
        """
        if isinstance(rec_hidden, tuple):
            h_n = RNNModule.get_last_hidden(rec_hidden[0], bidirect)
        else:
            if bidirect:
                h_n = torch.cat((rec_hidden[-2, :, :], rec_hidden[-1, :, :]), -1)
            else:
                h_n = rec_hidden[-1, :, :]
        return h_n

    @staticmethod
    def sort_tensors(input_tensor, input_lengths):
        """
        Sort input tensors by their lengths in a descending order
        """
        input_lengths, order_tensor = input_lengths.sort(0, descending=True)
        input_tensor = input_tensor[order_tensor]
        _, reorder_tensor = order_tensor.sort(0, descending=False)
        return input_tensor, input_lengths, order_tensor, reorder_tensor

    @staticmethod
    def resort_tensors(inp_tensor, reorder_tensor, dim=0):
        """
        Recover the original order of inp_tensor on dim dimension which orders are stored in reorder_tensor
        """
        if isinstance(inp_tensor, tuple):
            inp_tensor = tuple(RNNModule.resort_tensors(tensor, reorder_tensor, dim) for tensor in inp_tensor)
        else:
            num_dim = inp_tensor.dim()
            if reorder_tensor is not None:
                if dim == 0 and dim < num_dim:
                    if inp_tensor.size(0) != 1 and inp_tensor.size(0) == reorder_tensor.size(0):
                        inp_tensor = inp_tensor[reorder_tensor]
                elif dim == 1 and dim < num_dim:
                    if inp_tensor.size(1) != 1 and inp_tensor.size(1) == reorder_tensor.size(0):
                        inp_tensor = inp_tensor[:, reorder_tensor, :]
                elif dim == 2 and dim < num_dim:
                    if inp_tensor.size(2) != 1 and inp_tensor.size(2) == reorder_tensor.size(0):
                        inp_tensor = inp_tensor[:, :, reorder_tensor]
                else:
                    raise RuntimeError("Not implemented yet")
        return inp_tensor


class UniLSTMModel(RNNModule):
    """
     A Recurrent Model for classification with loss and inference functions
    """
    def __init__(self, rec_type, ntokens, emb_size, hidden_size, nlayers=1, dropout=0.5, bidirect=False, nlabels=5):
        super(UniLSTMModel, self).__init__(rec_type, ntokens, emb_size, hidden_size, nlayers, dropout, bidirect)
        # Create a FC layer for scoring
        self.scorer_layer = nn.Linear(hidden_size, nlabels)
        # Create a loss function
        # ignore_index = 0 to ignore calculating loss of PAD label
        self.celoss_layer = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

        hd_dim = hidden_size // 2 if bidirect else hidden_size
        self.num_labels = nlabels
        self.rec_type = rec_type
        self.nlayers = nlayers
        self.bidirect = bidirect
        self.hd_dim = hd_dim
        self.init_weights_scorer()

    def init_weights_scorer(self, initrange=0.1):
        """
        Initialize the FC layer randomly ranging between [-initrange, initrange]
        """
        self.scorer_layer.bias.data.zero_()
        self.scorer_layer.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        num_directions = 2 if self.bidirect else 1
        weight = next(self.parameters())
        if self.rec_type == 'LSTM':
            return (weight.new_zeros(self.nlayers*num_directions, bsz, self.hd_dim),
                    weight.new_zeros(self.nlayers*num_directions, bsz, self.hd_dim))
        else:
            return weight.new_zeros(self.nlayers*num_directions, bsz, self.hd_dim)

    def forward(self, input_tensor, input_lengths=None, init_hidden=None):
        """
        Override the forward function
        """
        #######################
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
        #######################

        # sort lengths of input tensors in the descending mode
        input_tensor, input_lengths, order_tensor, reorder_tensor = self.sort_tensors(input_tensor, input_lengths)
        # Project word embedding vectors [e1,...,en] from index input_tensor [id_w1,..., id_wn]
        emb = self.embedding_layer(input_tensor)
        # Dropout on embedding
        emb_drop = self.dropout_layer(emb)
        # Pass embedding vectors [e1,...,en]  into a recurrent model to learn hidden states [h1, ..., hn]
        rec_output, rec_hidden = self.get_all_hidden(emb_drop, input_lengths, init_hidden)
        # recover the original order of outputs to compute loss
        rec_output = self.resort_tensors(rec_output, reorder_tensor, dim=0)
        rec_hidden = self.resort_tensors(rec_hidden, reorder_tensor, dim=1)

        # Extract the last hidden vector
        last_hidden = UniLSTMModel.get_last_hidden(rec_hidden, self.bidirect)
        # Dropout on the last layer
        last_hidden_drop = self.dropout_layer(last_hidden)
        decoded_scores = self.scorer_layer(last_hidden_drop)
        return decoded_scores, rec_hidden, rec_output

    def NLL_loss(self, label_score, label_tensor):
        #######################
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
        #######################
        batch_loss = self.celoss_layer(label_score.contiguous().view(-1, self.num_labels),
                                       label_tensor.contiguous().view(-1, ))
        return batch_loss

    def inference(self, label_score, k=1):
        #######################
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
        #######################
        label_prob = F.softmax(label_score, dim=-1)
        label_prob, label_pred = label_prob.data.topk(k)
        return label_prob, label_pred


class BiLSTMModel(UniLSTMModel):
    def __init__(self, rec_type, ntokens, emb_size, hidden_size, nlayers=2, dropout=0.5, bidirect=True, nlabels=5):
        super(BiLSTMModel, self).__init__(rec_type, ntokens, emb_size, hidden_size, nlayers, dropout, bidirect, nlabels)
        # Create a FC layer for learning hidden representation
        self.fc_layer = nn.Linear(hidden_size, hidden_size)
        self.init_weights_fc()

    def init_weights_fc(self, initrange=0.1):
        """
        Initialize the FC layer randomly ranging between [-initrange, initrange]
        """
        self.fc_layer.bias.data.zero_()
        self.fc_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_tensor, input_lengths=None, init_hidden=None):
        """
        Override the forward function
        """
        #######################
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
        #######################

        # sort lengths of input tensors in the descending mode
        input_tensor, input_lengths, order_tensor, reorder_tensor = self.sort_tensors(input_tensor, input_lengths)
        # Project word embedding vectors [e1,...,en] from index input_tensor [id_w1,..., id_wn]
        emb = self.embedding_layer(input_tensor)
        # Dropout on embedding
        emb_drop = self.dropout_layer(emb)
        # Pass embedding vectors [e1,...,en]  into a recurrent model to learn hidden states [h1, ..., hn]
        rec_output, rec_hidden = self.get_all_hidden(emb_drop, input_lengths, init_hidden)
        # recover the original order of outputs to compute loss
        rec_output = self.resort_tensors(rec_output, reorder_tensor, dim=0)
        rec_hidden = self.resort_tensors(rec_hidden, reorder_tensor, dim=1)

        # Extract the last hidden vector
        last_hidden = BiLSTMModel.get_last_hidden(rec_hidden, self.bidirect)
        fc_hidden = self.fc_layer(last_hidden)
        # Dropout on the last layer
        fc_hidden_drop = self.dropout_layer(fc_hidden)
        decoded_scores = self.scorer_layer(fc_hidden_drop)
        return decoded_scores, rec_hidden, rec_output


if __name__ == '__main__':
    from data_utils import Vocab, Txtfile, Data2tensor, seqPAD, PAD
    cutoff = 5
    wl_th = -1
    batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_files = ["../dataset/train.small.txt"]
    vocab = Vocab(wl_th=wl_th, cutoff=cutoff)
    vocab.build(data_files, firstline=False)
    word2idx = vocab.wd2idx(vocab.w2i)
    label2idx = vocab.tag2idx(vocab.l2i)

    train_data = Txtfile(data_files[0], firstline=False, source2idx=word2idx, label2idx=label2idx)
    # train_data = [sent[0] for sent in train_data]
    train_batch = vocab.minibatches_with_label(train_data, batch_size=batch_size)
    inpdata=[]
    outdata=[]
    for doc, label in train_batch:
        doc_pad_ids, doc_lengths = seqPAD.pad_sequences(doc, pad_tok=vocab.w2i[PAD])
        doc_tensor = Data2tensor.idx2tensor(doc_pad_ids, device)
        doc_lengths_tensor = Data2tensor.idx2tensor(doc_lengths, device)
        label_tensor = Data2tensor.idx2tensor(label, device)
        inpdata.append(doc_tensor)
        outdata.append(label_tensor)
        break

    rec_type = "LSTM"
    ntoken = len(vocab.w2i)
    nlabels = len(vocab.l2i)
    emb_size = 32
    hidden_size = 64
    nlayers = 2
    dropout = 0.5
    bidirect = False

    # model = RNNModule(rec_type=rec_type, ntokens=ntoken, emb_size=emb_size, hidden_size=hidden_size, nlayers=nlayers,
    #                   dropout=dropout, bidirect=bidirect).to(device)
    # rec_output, rec_hidden, rec_output = model(input_tensor, input_lens_tensor)
    #
    # model = UniLSTMModel(rec_type=rec_type, ntokens=ntoken, emb_size=emb_size, hidden_size=hidden_size, nlayers=nlayers,
    #                      dropout=dropout, bidirect=False, nlabels=nlabels).to(device)
    # decoded_scores, rec_hidden, rec_output = model(input_tensor, input_lens_tensor)

    model = BiLSTMModel(rec_type=rec_type, ntokens=ntoken, emb_size=emb_size, hidden_size=hidden_size, nlayers=nlayers,
                        dropout=dropout, bidirect=True, nlabels=nlabels).to(device)

    decoded_scores, rec_hidden, rec_output = model(doc_tensor, doc_lengths_tensor)

    target = outdata[0]
    f_loss = model.NLL_loss(decoded_scores, target)

    label_prob, label_pred = model.inference(decoded_scores, k=1)


