import torch
import itertools
import gzip
import pickle
import random
import numpy as np
from collections import Counter

PAD = u"<PAD>"
UNK = u"<UNK>"
SOS = u"<s>"
EOS = u"</s>"


class Txtfile(object):
    """
    Read txt file
    """
    def __init__(self, fname, source2idx=None, label2idx=None, firstline=False, limit=-1):
        self.fname = fname
        self.firstline = firstline
        self.limit = limit if limit > 0 else None
        self.source2idx = source2idx
        self.label2idx = label2idx
        self.length = None

    def __iter__(self):
        with open(self.fname, 'r') as f:
            f.seek(0)
            if self.firstline:
                # Skip the header
                next(f)
            for line in itertools.islice(f, self.limit):
                source = line.strip()
                if len(source) != 0:
                    review, label = Txtfile.process_seq(source)
                    if self.source2idx is not None:
                        review = self.source2idx(review)
                    if self.label2idx is not None:
                        label = self.label2idx(label)
                    yield review, label

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length

    @staticmethod
    def process_seq(seq):
        seq = seq.lower().split()
        review = seq[1:]
        label = seq[0]
        return review, label


class Vocab(object):
    def __init__(self, wl_th=-1, cutoff=1):
        self.w2i = {}
        self.i2w = {}
        self.l2i = {}
        self.wl = wl_th
        self.cutoff = cutoff
                        
    def build(self, files, firstline=False, limit=-1):
        wl = 0
        sl = 0
        wcnt = Counter()
        lcnt = Counter()
        for fname in files:
            raw = Txtfile(fname, firstline=firstline, limit=limit)
            for sent, label in raw:
                wcnt.update(sent)
                lcnt.update([label])
                wl = max(wl, len(sent))
                sl += 1

        wlst = [x for x, y in wcnt.items() if y >= self.cutoff]
        wlst = [PAD, UNK, SOS, EOS] + wlst
        wvocab = dict([(y, x) for x, y in enumerate(wlst)])
        iwvocab = dict([(x, y) for x, y in enumerate(wlst)])

        lvocab = dict([(y[0], x) for x, y in enumerate(lcnt.most_common())])
        self.l2i = lvocab
        self.w2i = wvocab
        self.i2w = iwvocab
        self.wl = wl if self.wl <= 0 else min(wl, self.wl)

        print("Extracting vocabulary from %d total samples:" % sl)
        print("\t%d total labels, %d unique labels" % (sum(lcnt.values()), len(lcnt)))
        print("\t%d total tokens; %d unique tokens" % (sum(wcnt.values()), len(wcnt)))
        print("\t%d unique tokens appearing at least %d times" % (len(wvocab)-4, self.cutoff))

    @staticmethod
    def wd2idx(wvocab=None, allow_unk=True, start_end=True):
        """Return a function to convert tag2idx or word/char2idx"""
        def f(sent):                 
            if wvocab is not None:
                wd_ids = []
                for wd in sent:
                    # ignore words out of vocabulary
                    if wd in wvocab:
                        wd_ids += [wvocab[wd]]
                    else:
                        if allow_unk:
                            wd_ids += [wvocab[UNK]]
                        else:
                            print(wd)
                            raise Exception("Unknow key is not allowed. Check that "\
                                            "your vocab (tags?) is correct")  
                if start_end:
                    wd_ids = [wvocab[SOS]] + wd_ids + [wvocab[EOS]]
            return wd_ids
        return f

    @staticmethod
    def tag2idx(vocab_tags=None):
        def f(tag):
            return vocab_tags[tag]
        return f

    @staticmethod
    def minibatches(data, batch_size):
        """
        Args:
            data: generator of (sentence, tags) tuples
            minibatch_size: (int)
    
        Yields:
            list of tuples
    
        """
        x_batch = []
        for x, y in data:
            if len(x_batch) == batch_size:
                yield x_batch
                x_batch = []
            if type(x[0]) == tuple:
                x = list(zip(*x))
            x_batch += [x]
        if len(x_batch) > 0:
            yield x_batch
            
    @staticmethod
    def bptt_batch(tensor, i=0, bptt=10):
        seq_len = min(bptt, tensor.size(1) - 1 - i)
        data = tensor[:, i:i+seq_len]
        target = tensor[:, i+1:i+1+seq_len]
        return data, target


class Data2tensor:
    @staticmethod
    def idx2tensor(indexes, device=torch.device("cpu")):
        return torch.tensor(indexes, dtype=torch.long, device=device)

    @staticmethod
    def set_randseed(seed_num=12345):
        random.seed(seed_num)
        torch.manual_seed(seed_num)
        np.random.seed(seed_num)


class seqPAD:
    @staticmethod
    def _pad_sequences(sequences, pad_tok, max_length):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the word to pad with

        Returns:
            a list of list where each sublist has same length
        """
        sequence_padded, sequence_length = [], []

        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length

    @staticmethod
    def pad_sequences(sequences, pad_tok, nlevels=1, wthres=-1, cthres=-1):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the word to pad with
            nlevels: "depth" of padding, for the case where we have word ids

        Returns:
            a list of list where each sublist has same length

        """
        if nlevels == 1:
            max_length = max(map(lambda x: len(x), sequences))
            max_length = min(wthres, max_length) if wthres > 0 else max_length
            sequence_padded, sequence_length = seqPAD._pad_sequences(sequences, pad_tok, max_length)

        elif nlevels == 2:
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
            max_length_word = min(cthres, max_length_word) if cthres > 0 else max_length_word
            sequence_padded, sequence_length = [], []
            for seq in sequences:
                # pad the word-level first to make the word length being the same
                sp, sl = seqPAD._pad_sequences(seq, pad_tok, max_length_word)
                sequence_padded += [sp]
                sequence_length += [sl]
            # pad the word-level to make the sequence length being the same
            max_length_sentence = max(map(lambda x: len(x), sequences))
            max_length_sentence = min(wthres, max_length_sentence) if wthres > 0 else max_length_sentence
            sequence_padded, _ = seqPAD._pad_sequences(sequence_padded, [pad_tok] * max_length_word,
                                                       max_length_sentence)
            # set sequence length to 1 by inserting padding
            sequence_length, _ = seqPAD._pad_sequences(sequence_length, 1, max_length_sentence)

        return sequence_padded, sequence_length


# Save and load hyper-parameters
class SaveloadHP:
    @staticmethod            
    def save(args, argfile='./results/model_args.pklz'):
        """
        argfile='model_args.pklz'
        """
        print("Writing hyper-parameters into %s" % argfile)
        with gzip.open(argfile, "wb") as fout:
            pickle.dump(args, fout, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(argfile='./results/model_args.pklz'):
        print("Reading hyper-parameters from %s" % argfile)
        with gzip.open(argfile, "rb") as fin:
            args = pickle.load(fin)
        return args

    
if __name__ == '__main__':
    cutoff = 5
    wl_th = -1
    batch_size = 16
    bptt = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    data_files = ["../dataset/train.small.txt"]
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
