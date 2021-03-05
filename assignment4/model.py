# coding: utf-8
import argparse
import time
import math
import torch
import os
import torch.onnx
import torch.optim as optim
from sklearn import metrics
from utils.data_utils import Vocab, Txtfile, Data2tensor, SaveloadHP, seqPAD, PAD
from utils.core_nns import UniLSTMModel, BiLSTMModel

# Set the random seed manually for reproducibility.
Data2tensor.set_randseed(1234)


class Sentimentmodel(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if self.args.use_cuda else "cpu")
        self.word2idx = self.args.vocab.wd2idx(self.args.vocab.w2i,
                                               allow_unk=self.args.allow_unk, start_end=self.args.se_words)
        self.label2idx = self.args.vocab.tag2idx(self.args.vocab.l2i)
        self.ntokens = len(self.args.vocab.w2i)
        self.nlabels = len(self.args.vocab.l2i)
        if args.bidirect:
            self.model = BiLSTMModel(args.model, self.ntokens, args.emb_size, args.hidden_size, args.nlayers, args.dropout,
                                     args.bidirect, nlabels=self.nlabels).to(self.device)
        else:
            self.model = UniLSTMModel(args.model, self.ntokens, args.emb_size, args.hidden_size, args.nlayers, args.dropout,
                                      args.bidirect, nlabels=self.nlabels).to(self.device)

        self.model_optimizer = None
        if self.args.optimizer_type.lower() == "adamax":
            self.init_optimizers(optim.Adamax)
        elif self.args.optimizer_type.lower() == "adam":
            self.init_optimizers(optim.Adam)
        elif self.args.optimizer_type.lower() == "adadelta":
            self.init_optimizers(optim.Adadelta)
        elif self.args.optimizer_type.lower() == "adagrad":
            self.init_optimizers(optim.Adagrad)
        else:
            self.init_optimizers(optim.SGD)

    def init_optimizers(self, opt_method=optim.SGD):
        self.model_optimizer = opt_method(self.model.parameters(), lr=self.args.lr)

    @staticmethod
    def cal_metrics(y_true, y_pred):
        acc = metrics.accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted')
        return precision, recall, f1, acc

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

    def evaluate_batch(self, eval_data):
        start_time = time.time()
        eval_batch = self.args.vocab.minibatches_with_label(eval_data, batch_size=self.args.batch_size)
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0.
        total_docs = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for doc_batch, lb_batch in eval_batch:
                doc_pad_ids, doc_lengths = seqPAD.pad_sequences(doc_batch, pad_tok=self.args.vocab.w2i[PAD])
                #######################
                # YOUR CODE STARTS HERE

                # YOUR CODE ENDS HERE
                #######################

        precision, recall, f1, acc = Sentimentmodel.cal_metrics(y_true, y_pred)
        cur_loss = total_loss / total_docs
        elapsed = time.time() - start_time
        metrics = {"precision": precision * 100,
                   "recall": recall * 100,
                   "f1": f1 * 100,
                   "acc": acc * 100,
                   "loss": cur_loss
                   }
        return metrics, total_docs, elapsed

    def train_batch(self, train_data):
        total_loss = 0.
        total_docs = 0
        start_time = time.time()
        train_batch = self.args.vocab.minibatches_with_label(train_data, batch_size=self.args.batch_size)
        # Turn on training mode which enables dropout.
        self.model.train()
        for batch, (doc_batch, lb_batch) in enumerate(train_batch):
            doc_pad_ids, doc_lengths = seqPAD.pad_sequences(doc_batch, pad_tok=self.args.vocab.w2i[PAD])
            doc_tensor = Data2tensor.idx2tensor(doc_pad_ids, self.device)
            doc_lengths_tensor = Data2tensor.idx2tensor(doc_lengths, self.device)
            lb_tensor = Data2tensor.idx2tensor(lb_batch, self.device)
            # doc_tensor = [batch_size, max_doc_length]
            total_docs += doc_tensor.size(0)

            self.model.zero_grad()
            output, _, _ = self.model(doc_tensor, doc_lengths_tensor)
            loss = self.model.NLL_loss(output, lb_tensor)
            avg_loss = loss/doc_tensor.size(0)
            avg_loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

            # update parameters in all sub-graphs
            self.model_optimizer.step()
            # for p in self.model.parameters():
            #     p.data.add_(p.grad.data, alpha=-self.args.lr)

            total_loss += loss.item()

        cur_loss = total_loss / total_docs
        elapsed = time.time() - start_time
        # print('-' * 89)
        # print('| TRAINING | epoch {:3d} | documents {:5d} | lr {:02.2f} | documents/s {:5.2f} | '
        #       'loss {:5.2f}'.format(epoch, total_docs, self.args.lr, total_docs / elapsed, cur_loss))
        # print('-' * 89)
        return cur_loss, total_docs, elapsed

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
                epoch_start_train_time = time.time()
                train_loss, train_total_docs, train_elapsed = self.train_batch(train_data)
                print('-' * 89)
                print('|  EPOCH {:d}: TRAINING   | time: {:5.2f}s | loss {:5.2f} | documents {:5d} | '
                      'documents/s {:5.2f} |'.format(epoch, (time.time() - epoch_start_train_time), train_loss,
                                                     train_total_docs, train_total_docs / train_elapsed))

                epoch_start_eval_time = time.time()
                val_metrics, val_total_docs, val_elapsed = self.evaluate_batch(dev_data)
                print('|  EPOCH {:d}: EVALUATING | time: {:5.2f}s | loss {:5.2f} | f1 {:5.2f} | documents {:5d} | '
                      'documents/s {:5.2f} |'.format(epoch, (time.time() - epoch_start_eval_time), val_metrics["loss"],
                                                     val_metrics["f1"], val_total_docs, val_elapsed))

                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_metrics["loss"] < best_val_loss:
                    nimp = 0
                    best_epoch = epoch
                    # save the model into file
                    # Convert model to CPU to avoid out of GPU memory
                    self.model.to("cpu")
                    torch.save(self.model.state_dict(), self.args.trained_model)
                    self.model.to(self.device)

                    best_val_loss = val_metrics["loss"]

                    print('|          ---------> NEW IMPROVEMENT ---------> Save the model to file')
                    print('-' * 89)

                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    # self.args.lr /= 2.0
                    nimp += 1
                    if nimp >= self.args.es:
                        # load the model for testing
                        self.model.load_state_dict(torch.load(self.args.trained_model))
                        self.model.to(self.device)
                        epoch_start_test_time = time.time()
                        test_metrics, test_total_docs, test_elapsed = self.evaluate_batch(test_data)
                        print("|          ---------> EARLY STOPPING  at epoch {:3d}".format(epoch))

                        print('|  BEST EPOCH {:d}: TESTING | time: {:5.2f}s | loss {:5.2f} | f1 {:5.2f} | '
                              'documents {:5d} | documents/s {:5.2f} |'.format(best_epoch,
                                                                               (time.time() - epoch_start_test_time),
                                                                               test_metrics["loss"], test_metrics["f1"],
                                                                               test_total_docs, test_elapsed))
                        print('-' * 89)
                    else:
                        print('-' * 89)

            # load the model for testing
            self.model.load_state_dict(torch.load(self.args.trained_model))
            self.model.to(self.device)
            epoch_start_test_time = time.time()
            test_metrics, test_total_docs, test_elapsed = self.evaluate_batch(test_data)
            print('|  BEST EPOCH {:3d}: TESTING | time: {:5.2f}s | loss {:5.2f} | f1 {:5.2f} | documents {:5d} | '
                  'documents/s {:5.2f} |'.format(best_epoch, (time.time() - epoch_start_test_time),
                                                 test_metrics["loss"],  test_metrics["f1"],
                                                 test_total_docs, test_elapsed))
            print('-' * 89)

        except KeyboardInterrupt:
            print('-' * 89)
            print('|         ---------> KEY INTERRUPTION ---------> Exiting from training early')
            # load the model for testing
            self.model.load_state_dict(torch.load(self.args.trained_model))
            self.model.to(self.device)
            epoch_start_test_time = time.time()
            test_metrics, test_total_docs, test_elapsed = self.evaluate_batch(test_data)
            print('|  BEST EPOCH {:3d}: TESTING | time: {:5.2f}s | loss {:5.2f} | f1 {:5.2f} | documents {:5d} | '
                  'documents/s {:5.2f} |'.format(best_epoch, (time.time() - epoch_start_test_time),
                                                 test_metrics["loss"], test_metrics["f1"],
                                                 test_total_docs, test_elapsed))
            print('-' * 89)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language Model')
    
    parser.add_argument('--train_file', help='Trained file', default="./dataset/train.small.txt", type=str)

    parser.add_argument('--dev_file', help='Trained file', default="./dataset/val.small.txt", type=str)
    
    parser.add_argument('--test_file', help='Tested file', default="./dataset/test.small.txt", type=str)

    parser.add_argument("--wl_th", type=int, default=-1, help="Word threshold")

    parser.add_argument("--cutoff", type=int, default=1, help="Prune words occurring <= wcutoff")

    parser.add_argument("--se_words", action='store_false', default=True, help="Start-end padding flag at word")

    parser.add_argument("--allow_unk", action='store_false', default=True, help="allow using unknown padding")
        
    parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')

    parser.add_argument('--emb_size', type=int, default=50, help='size of word embeddings')
    
    parser.add_argument('--hidden_size', type=int, default=50, help='number of hidden units per layer')
    
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    
    parser.add_argument('--bidirect', action='store_false', default=True, help='bidirectional flag')

    parser.add_argument("--optimizer_type", default="adamw", type=str, help="An optimizer method", )

    parser.add_argument("--lr", default=1e-3, type=float, help="The initial learning rate.")
    
    parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
    
    parser.add_argument('--epochs', type=int, default=8, help='upper epoch limit')

    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size')

    parser.add_argument('--es', type=int, default=2, help='Early stopping criterion')
    
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout)')

    parser.add_argument('--model_dir', type=str, default='./results/', help='directory to store trained models')

    parser.add_argument('--trained_model', type=str, default='./results/model.m', help='path to save the final model')

    parser.add_argument('--model_args', type=str, default='./results/argument.args',
                        help='path to save the model argument')

    parser.add_argument("--use_cuda", action='store_true', default=False, help="GPUs Flag (default False)")
        
    args = parser.parse_args()

    prefix = "senti_cls_uni"
    if args.bidirect:
        prefix = "senti_cls_bi"

    args.trained_model = os.path.join(args.model_dir, prefix + args.model.lower()+".m")
    args.model_args = os.path.join(args.model_dir, prefix + args.model.lower() + ".args")
    
    args = Sentimentmodel.build_data(args)
    
    sentiment_model = Sentimentmodel(args)
    
    sentiment_model.train()
