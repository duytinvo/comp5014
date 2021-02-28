# coding: utf-8
import argparse
import time
import math
import torch
import os
import torch.onnx
from utils.data_utils import Vocab, Txtfile, Data2tensor, SaveloadHP, seqPAD, PAD
from utils.core_nns import RNNModel

# Set the random seed manually for reproducibility.
Data2tensor.set_randseed(1234)


class Languagemodel(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if self.args.use_cuda else "cpu")
        self.word2idx = self.args.vocab.wd2idx(self.args.vocab.w2i,
                                               allow_unk=self.args.allow_unk, start_end=self.args.se_words)
        self.label2idx = self.args.vocab.tag2idx(self.args.vocab.l2i)
        self.ntokens = len(self.args.vocab.w2i)
        self.model = RNNModel(args.model, self.ntokens, args.emsize, args.nhid,
                              args.nlayers, args.dropout, args.tied, args.bidirect).to(self.device)

    def repackage_hidden(self, h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def bptt_batch(self, source, i):
        seq_len = min(self.args.bptt, source.size(1) - 1 - i)
        data = source[:, i:i+seq_len]
        target = source[:, i+1:i+1+seq_len]
        return data, target
    
    def evaluate_batch(self, eval_data):
        start_time = time.time()
        eval_batch = self.args.vocab.minibatches(eval_data, batch_size=self.args.batch_size)
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0.
        total_word = 0
        with torch.no_grad():
            for seq_batch in eval_batch:
                word_pad_ids, seq_lens = seqPAD.pad_sequences(seq_batch, pad_tok=self.args.vocab.w2i[PAD])
                seq_tensor = Data2tensor.idx2tensor(word_pad_ids, self.device)
                hidden = self.model.init_hidden(seq_tensor.size(0))
                for i in range(0, seq_tensor.size(1) - 1, self.args.bptt):
                    data, target = self.bptt_batch(seq_tensor, i)
                    mask_target = target > 0
                    output, hidden = self.model(data, hidden)
                    batch_loss = self.model.NLL_loss(output, target)
                    total_loss += batch_loss.item()
                    hidden = self.repackage_hidden(hidden)
                    total_word = total_word + mask_target.sum().item()

        cur_loss = total_loss / total_word
        elapsed = time.time() - start_time
        print('-' * 89)
        print('| EVALUATION | words {:5d} | lr {:02.2f} | words/s {:5.2f} | '
              'loss {:5.2f} | ppl {:8.2f}'.format(total_word, self.args.lr,
                                                  total_word / elapsed, cur_loss, math.exp(cur_loss)))
        print('-' * 89)
        return cur_loss, total_word, elapsed

    def train_batch(self, train_data, epoch=0):
        total_loss = 0.
        total_word = 0
        total_seq = 0
        start_time = time.time()
        train_batch = self.args.vocab.minibatches(train_data, batch_size=self.args.batch_size)
        # Turn on training mode which enables dropout.
        self.model.train()
        for batch, seq_batch in enumerate(train_batch):
            word_pad_ids, seq_lens = seqPAD.pad_sequences(seq_batch, pad_tok=self.args.vocab.w2i[PAD])
            seq_tensor = Data2tensor.idx2tensor(word_pad_ids, self.device)
            # seq_tensor = [batch_size, seq_len]
            total_seq += seq_tensor.size(0)
            hidden = self.model.init_hidden(seq_tensor.size(0))
            for i in range(0, seq_tensor.size(1)-1, self.args.bptt):
                # data = [batch_size, bptt]
                # target = [batch_size, bptt]
                data, target = self.bptt_batch(seq_tensor, i)
                mask_target = target > 0
                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden = self.repackage_hidden(hidden)
                self.model.zero_grad()
                output, hidden = self.model(data, hidden)
                loss = self.model.NLL_loss(output, target)
                loss.backward()
        
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                for p in self.model.parameters():
                    p.data.add_(-self.args.lr, p.grad.data)
        
                total_loss += loss.item()
                total_word = total_word + mask_target.sum().item()
                
            cur_loss = total_loss / total_word
            elapsed = time.time() - start_time
            print('-' * 89)
            print('| TRAINING | epoch {:3d} | batch {:5d} | sequences {:5d} | words {:5d} | lr {:02.2f} | '
                  'words/s {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch+1, total_seq, total_word,
                                                                        self.args.lr, total_word/elapsed, cur_loss,
                                                                        math.exp(cur_loss)))
            print('-' * 89)
            
    def train(self):
        train_data = Txtfile(self.args.train_file, firstline=False, source2idx=self.word2idx, label2idx=self.label2idx)
        dev_data = Txtfile(self.args.dev_file, firstline=False, source2idx=self.word2idx, label2idx=self.label2idx)
        test_data = Txtfile(self.args.test_file, firstline=False, source2idx=self.word2idx, label2idx=self.label2idx)
        best_val_loss = None
        best_epoch = 0
        nimp = 0
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            # Loop over epochs.
            for epoch in range(1, self.args.epochs + 1):
                epoch_start_time = time.time()
                self.train_batch(train_data, epoch)
                val_loss, val_total_word, val_elapsed = self.evaluate_batch(dev_data)
                print('-' * 89)
                print('| EVALUATING | end of epoch {:3d} | time: {:5.2f}s | val loss {:5.2f} | val ppl {:8.2f} | '
                      'val_word {:5d} | words/s {:5.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                                val_loss, math.exp(val_loss), val_total_word,
                                                                val_elapsed))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    nimp = 0
                    best_epoch = epoch
                    # save the model into file
                    # Convert model to CPU to avoid out of GPU memory
                    self.model.to("cpu")
                    torch.save(self.model.state_dict(), self.args.trained_model)
                    self.model.to(self.device)

                    best_val_loss = val_loss

                    print('-' * 89)
                    print('| NEW IMPROVEMENT | Save the model to file')
                    print('-' * 89)
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    self.args.lr /= 4.0
                    nimp += 1
                    if nimp >= self.args.es:
                        # load the model for testing
                        self.model.load_state_dict(torch.load(self.args.trained_model))
                        self.model.to(self.device)

                        test_loss, test_total_word, test_elapsed = self.evaluate_batch(test_data)
                        print('-' * 89)
                        print("Early Stopping at epoch {:3d}".format(epoch))
                        print(
                            '| TESTING | best epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | test ppl {:8.2f} | '
                            'test_word {:5d} | words/s {:5.2f}'.format(best_epoch, (time.time() - epoch_start_time),
                                                                       test_loss, math.exp(test_loss), test_total_word,
                                                                       test_elapsed))
                        print('-' * 89)
                        return

            # load the model for testing
            self.model.load_state_dict(torch.load(self.args.trained_model))
            self.model.to(self.device)

            test_loss, test_total_word, test_elapsed = self.evaluate_batch(test_data)
            print('-' * 89)
            print(
                '| TESTING | best epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | test ppl {:8.2f} | '
                'test_word {:5d} | words/s {:5.2f}'.format(best_epoch, (time.time() - epoch_start_time),
                                                           test_loss, math.exp(test_loss), test_total_word,
                                                           test_elapsed))
            print('-' * 89)
                     
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
            # load the model for testing
            self.model.load_state_dict(torch.load(self.args.trained_model))
            self.model.to(self.device)

            test_loss, test_total_word, test_elapsed = self.evaluate_batch(test_data)
            print('-' * 89)
            print(
                '| TESTING | best epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | test ppl {:8.2f} | '
                'test_word {:5d} | words/s {:5.2f}'.format(best_epoch, (time.time() - epoch_start_time),
                                                           test_loss, math.exp(test_loss), test_total_word,
                                                           test_elapsed))
            print('-' * 89)

    @staticmethod
    def build_data(args):
        print("Building dataset...")
        model_dir, _ = os.path.split(args.model_args)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        vocab = Vocab(wl_th=args.wl_th, cutoff=args.cutoff)
        vocab.build([args.train_file, args.dev_file], firstline=False)
        args.vocab = vocab
        SaveloadHP.save(args, args.model_args)
        return args
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language Model')
    
    parser.add_argument('--train_file', help='Trained file', default="./dataset/train.small.txt", type=str)

    parser.add_argument('--dev_file', help='Trained file', default="./dataset/val.small.txt", type=str)
    
    parser.add_argument('--test_file', help='Tested file', default="./dataset/test.small.txt", type=str)

    parser.add_argument("--wl_th", type=int, default=-1, help="Word threshold")

    parser.add_argument("--cutoff", type=int, default=1, help="Prune words occurring <= wcutoff")

    parser.add_argument("--se_words", action='store_false', default=True, help="Start-end padding flag at word")

    parser.add_argument("--allow_unk", action='store_false', default=True, help="allow using unknown padding")
        
    parser.add_argument('--model', type=str, default='RNN_TANH', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')

    parser.add_argument('--emsize', type=int, default=16, help='size of word embeddings')
    
    parser.add_argument('--nhid', type=int, default=32, help='number of hidden units per layer')
    
    parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
    
    parser.add_argument('--bidirect', action='store_true', default=False, help='bidirectional flag')
    
    parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
    
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    
    parser.add_argument('--epochs', type=int, default=8, help='upper epoch limit')
    
    # batch_size: here, it is the number of sentence being training at the same time
    # The actual value of batch_size = batch_size * bptt
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size')
        
    parser.add_argument('--bptt', type=int, default=32, help='sequence length')
    
    parser.add_argument('--es', type=int, default=2, help='Early stopping criterion')
    
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout)')
    
    parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
            
    parser.add_argument('--trained_model', type=str, default='./results/lm.m', help='path to save the final model')

    parser.add_argument('--model_args', type=str, default='./results/lm.args', help='path to save the model argument')

    parser.add_argument("--use_cuda", action='store_true', default=False, help="GPUs Flag (default False)")
        
    args = parser.parse_args()
    
    args = Languagemodel.build_data(args)
    
    lm = Languagemodel(args)
    
    lm.train()
