# data stream
import logging
import cPickle as pkl
import os
import numpy
from fuel.datasets import TextFile
from fuel.streams import DataStream


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DStream(object):

    def __init__(self, **kwards):

        self.train_src = kwards.pop('train_src')
        self.train_trg = kwards.pop('train_trg')
        self.vocab_src = kwards.pop('vocab_src')
        self.vocab_trg = kwards.pop('vocab_trg')
        self.unk_token = kwards.pop('unk_token')
        self.unk_id = kwards.pop('unk_id')
        self.eos_token = kwards.pop('eos_token')
        self.src_vocab_size = kwards.pop('src_vocab_size')
        self.trg_vocab_size = kwards.pop('trg_vocab_size')
        self.seq_len_src = kwards.pop('seq_len_src')
        self.seq_len_trg = kwards.pop('seq_len_trg')
        self.batch_size = kwards.pop('batch_size')
        self.sort_k_batches = kwards.pop('sort_k_batches')

        # get source and target dicts
        self.src_dict, self.trg_dict = self._get_dict()
        self.eos_id = self.src_dict[self.eos_token]

        # convert senteces to ids and filter length > seq_len_src / seq_len_trg
        self.source, self.target = self._get_sentence_pairs()

        # sorted k batches
        if self.sort_k_batches > 1:
            self.source, self.target = self._sort_by_k_batches(self.source, self.target)

        num_sents = len(self.source)
        assert num_sents == len(self.target)

        if num_sents % self.batch_size == 0:
            self.blocks = num_sents / self.batch_size
        else:
            self.blocks = num_sents / self.batch_size + 1


    def get_iterator(self):

        for i in range(self.blocks):
            x = self.source[i*self.batch_size: (i+1)*self.batch_size]
            y = self.target[i*self.batch_size: (i+1)*self.batch_size]
            batch = self._create_padded_batch(x, y)
            yield batch


    def _create_padded_batch(self, x, y):

        mx = numpy.minimum(self.seq_len_src, max([len(xx) for xx in x])) + 1
        my = numpy.minimum(self.seq_len_trg, max([len(xx) for xx in y])) + 1

        batch_size = len(x)

        X = numpy.zeros((batch_size, mx), dtype='int64')
        Y = numpy.zeros((batch_size, my), dtype='int64')
        Xmask = numpy.zeros((batch_size, mx), dtype='float32')
        Ymask = numpy.zeros((batch_size, my), dtype='float32')

        for idx in range(len(x)):
            X[idx, :len(x[idx])] = x[idx]
            Xmask[idx, :len(x[idx])] = 1.
            if len(x[idx]) < mx:
                X[idx, len(x[idx]):] = self.eos_id
                Xmask[idx, len(x[idx])] = 1.

        for idx in range(len(y)):
            Y[idx,:len(y[idx])] = y[idx]
            Ymask[idx,:len(y[idx])] = 1.
            if len(y[idx]) < my:
                Y[idx, len(y[idx]):] = self.eos_id
                Ymask[idx, len(y[idx])] = 1.

        return X, Xmask, Y, Ymask


    def _get_dict(self):

        if os.path.isfile(self.vocab_src):
            src_dict = pkl.load(open(self.vocab_src, 'rb'))
        else:
            logger.error("file [{}] do not exist".format(self.vocab_src))

        if os.path.isfile(self.vocab_trg):
            trg_dict = pkl.load(open(self.vocab_trg, 'rb'))
        else:
            logger.error("file [{}] do not exist".format(self.vocab_trg))

        return src_dict, trg_dict


    def _get_sentence_pairs(self):

        if os.path.isfile(self.train_src):
            f_src = open(self.train_src, 'r')
        else:
            logger.error("file [{}] do not exist".format(self.train_src))

        if os.path.isfile(self.train_trg):
            f_trg = open(self.train_trg, 'r')
        else:
            logger.error("file [{}] do not exist".format(self.train_trg))


        source = []
        target = []

        for l_src, l_trg in zip(f_src, f_trg):
            src_words = l_src.strip().split()
            #src_words.append(self.eos_token)

            trg_words = l_trg.strip().split()
            #trg_words.append(self.eos_token)

            if len(src_words) == 0 or len(trg_words) == 0:
                continue

            if len(src_words) > self.seq_len_src or len(trg_words) > self.seq_len_trg:
                continue

            src_ids = [self.src_dict[w] if w in self.src_dict else self.unk_id for w in src_words]
            trg_ids = [self.trg_dict[w] if w in self.trg_dict else self.unk_id for w in trg_words]

            source.append(src_ids)
            target.append(trg_ids)

        f_src.close()
        f_trg.close()

        return source, target


    def _sort_by_k_batches(self, source, target):

        bs = self.batch_size * self.sort_k_batches
        num_sents = len(source)
        assert num_sents == len(target)

        if num_sents % bs == 0:
            blocks = num_sents / bs
        else:
            blocks = num_sents / bs + 1

        sort_source = []
        sort_target = []
        for i in range(blocks):
            tmp_src = numpy.asarray(source[i*bs:(i+1)*bs])
            tmp_trg = numpy.asarray(target[i*bs:(i+1)*bs])
            lens = numpy.asarray([map(len, tmp_src), map(len, tmp_trg)])
            orders = numpy.argsort(lens[-1])
            for idx in orders:
                sort_source.append(tmp_src[idx])
                sort_target.append(tmp_trg[idx])

        return sort_source, sort_target


def get_devtest_stream(data_type='valid', input_file=None, **kwards):

    if data_type == 'valid':
        data_file = kwards.pop('valid_src')
    elif data_type == 'test':
        if input_file is None:
            data_file = kwards.pop('test_src')
        else:
            data_file = input_file
    else:
        logger.error('wrong datatype, which must be one of valid or test')

    unk_token = kwards.pop('unk_token')
    eos_token = kwards.pop('eos_token')
    vocab_src = kwards.pop('vocab_src')

    dataset = TextFile(files=[data_file],
                       dictionary=pkl.load(open(vocab_src, 'rb')),
                       level='word',
                       unk_token=unk_token,
                       bos_token=None,
                       eos_token=eos_token)

    dev_stream = DataStream(dataset)

    return dev_stream


# added by Zhaopeng Tu
def get_stream(input_file, vocab_file, **kwards):
    unk_token = kwards.pop('unk_token')
    eos_token = kwards.pop('eos_token')

    dataset = TextFile(files=[input_file],
                       dictionary=pkl.load(open(vocab_file, 'rb')),
                       level='word',
                       unk_token=unk_token,
                       bos_token=None,
                       eos_token=eos_token)

    stream = DataStream(dataset)

    return stream

