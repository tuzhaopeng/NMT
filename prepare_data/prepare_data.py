# prepare data especially the dictionary
import logging
import argparse
import pprint
import os
import cPickle as pkl
from collections import Counter
import configurations


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--proto",  default="get_config_search_coverage",
                     help="Prototype config to use for config")
args = parser.parse_args()

class PrepareData(object):

    def __init__(self, **kwards):

        self.train_src = kwards.pop('train_src')
        self.train_trg = kwards.pop('train_trg')
        self.vocab_src = kwards.pop('vocab_src')
        self.vocab_trg = kwards.pop('vocab_trg')
        self.unk_token = kwards.pop('unk_token')
        self.bos_token = kwards.pop('bos_token')
        self.eos_token = kwards.pop('eos_token')
        self.src_vocab_size = kwards.pop('src_vocab_size')
        self.trg_vocab_size = kwards.pop('trg_vocab_size')
        self.unk_id = kwards.pop('unk_id')
        self.bos_id = 0
        self.eos_id = 0
        self.seq_len_src = kwards.pop('seq_len_src')
        self.seq_len_trg = kwards.pop('seq_len_trg')

        src_dict, trg_dict = self._create_dictionary()
        
        with open(self.vocab_src, 'wb') as f:
            pkl.dump(src_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.close()
        with open(self.vocab_trg, 'wb') as f:
            pkl.dump(trg_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.close()

        logger.info('vocab [{}] and [{}] has been dumped'.format(self.vocab_src, self.vocab_trg))


    def _create_dictionary(self):

        # Part 0: Read corpora
        if os.path.isfile(self.train_src):
            f_src = open(self.train_src, 'r')
        else:
            logger.error("file [{}] do not exist".format(self.train_src))

        if os.path.isfile(self.train_trg):
            f_trg = open(self.train_trg, 'r')
        else:
            logger.error("file [{}] do not exist".format(self.train_trg))

        sentences_src = f_src.readlines()
        sentences_trg = f_trg.readlines()

        f_src.close()
        f_trg.close()

        print len(sentences_src), '\t', len(sentences_trg)
        assert len(sentences_src) == len(sentences_trg)

        # Part I: Counting the words
        counter_src = Counter()
        counter_trg = Counter()
        for line_src, line_trg in zip(sentences_src, sentences_trg):
            words_src = line_src.strip().split()
            words_trg = line_trg.strip().split()

            if len(words_src) == 0 or len(words_trg) == 0:
                continue

            if self.seq_len_src < len(words_src) or self.seq_len_trg < len(words_trg):
                continue

            counter_src.update(words_src)
            counter_trg.update(words_trg)

        logger.info("Source Total: %d unique words, with a total of %d words."
                % (len(counter_src), sum(counter_src.values())))

        logger.info("Target Total: %d unique words, with a total of %d words."
                % (len(counter_trg), sum(counter_trg.values())))

        # Part II: Creating the dictionary
        special_tokens = [self.unk_token, self.bos_token, self.eos_token]
        for st in special_tokens:
            if st in counter_src:
                del counter_src[st]
            if st in counter_trg:
                del counter_trg[st]

        if self.src_vocab_size < 2:
            self.src_vocab_size = len(counter_src) + 2
        if self.trg_vocab_size < 2:
            self.trg_vocab_size = len(counter_trg) + 2

        src_valid_count = counter_src.most_common(self.src_vocab_size - 2)
        src_dict = {self.bos_token:self.bos_id, self.eos_token:self.eos_id, self.unk_token:self.unk_id}
        src_word_counts = 0
        for i, (word, count) in enumerate(src_valid_count):
            src_dict[word] = i + 2
            src_word_counts += count
        logger.info('Source dict contains %d words, covers %.1f%% of the text'
                        %(self.src_vocab_size, 100.0*src_word_counts/sum(counter_src.values())))

        trg_valid_count = counter_trg.most_common(self.trg_vocab_size - 2)
        trg_dict = {self.bos_token:self.bos_id, self.eos_token:self.eos_id, self.unk_token:self.unk_id}
        trg_word_counts = 0
        for i, (word, count) in enumerate(trg_valid_count):
            trg_dict[word] = i + 2
            trg_word_counts += count
        logger.info('Target dict contains %d words, covering %.1f%% of the text'
                        %(self.trg_vocab_size, 100.0*trg_word_counts/sum(counter_trg.values())))

        return src_dict, trg_dict


if __name__ == '__main__':
    configuration = getattr(configurations, args.proto)()
    logger.info("\nModel options:\n{}".format(pprint.pformat(configuration)))
    PrepareData(**configuration)

