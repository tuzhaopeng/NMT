# sampling: Sampler and BleuValidator
#from __future__ import print_function
import numpy
import argparse
import pprint
import os
import cPickle as pkl
import subprocess
import logging
import time
import re
import configurations
from search import BeamSearch
from nmt import EncoderDecoder
from stream import get_devtest_stream
# added by Zhaopeng Tu, 2016-07-21
from pinyin import get_pinyin

logger = logging.getLogger(__name__)


class Sampler(object):

    def __init__(self, search_model, **kwards):

        self.search_model = search_model
        self.unk_token = kwards.pop('unk_token')
        self.eos_token = kwards.pop('eos_token')
        self.vocab_src = kwards.pop('vocab_src')
        self.vocab_trg = kwards.pop('vocab_trg')
        self.hook_samples = kwards.pop('hook_samples')
       
        # added by Zhaopeng Tu, 2016-05-04
        self.with_coverage = kwards.pop('with_coverage')
        self.coverage_type = kwards.pop('coverage_type')

        self.dict_src, self.idict_src = self._get_dict(self.vocab_src)
        self.dict_trg, self.idict_trg = self._get_dict(self.vocab_trg)


    def apply(self, src_batch, trg_batch):

        batch_size = src_batch.shape[0]
        hook_samples = min(batch_size, self.hook_samples)
        sample_idx = numpy.random.choice(batch_size, hook_samples, replace=False)
        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]

        for i in range(hook_samples):
            input_length = self._get_true_length(input_[i], self.dict_src)
            target_length = self._get_true_length(target_[i], self.dict_trg)

            inp = input_[i, :input_length]
            # modified by Zhaopeng Tu, 2016-05-04
            # outputs, costs = self.search_model.apply(inp[:, None])
            results = self.search_model.apply(inp[:, None])
            outputs, costs, alignments = results[:3]
            idx = 3
            if self.with_coverage:
                coverages = results[idx]
                idx += 1
                if self.coverage_type is 'linguistic':
                    fertilities = results[idx]
                    idx += 1
            
            sample_length = self._get_true_length(numpy.array(outputs), self.dict_trg)

            logger.info("Input: {}".format(self._idx_to_word(input_[i][:input_length], self.idict_src)))
            logger.info("Target: {}".format(self._idx_to_word(target_[i][:target_length], self.idict_trg)))
            logger.info("Output: {}".format(self._idx_to_word(outputs[:sample_length], self.idict_trg)))
            # added by Zhaopeng Tu, 2016-05-04
            if self.with_coverage:
                logger.info("Coverage: {}".format(self._idx_to_word(input_[i][:input_length], self.idict_src, coverages)))
                if self.coverage_type is 'linguistic':
                    logger.info("Fertility: {}".format(self._idx_to_word(input_[i][:input_length], self.idict_src, fertilities)))
            logger.info("Cost: %.4f\n" %costs)

    
    def _get_dict(self, vocab_file):

        if os.path.isfile(vocab_file):
            ddict = pkl.load(open(vocab_file, 'rb'))
        else:
            logger.error("file [{}] do not exist".format(vocab_file))

        iddict = dict()
        for kk, vv in ddict.iteritems():
            iddict[vv] = kk

        iddict[0] = self.eos_token
        # <UNK> bug, it will be tokenized into '<', 'UNK', '>' by mteval.perl
        iddict[1] = 'UNK'

        return ddict, iddict


    def _get_true_length(self, seq, vocab):

        try:
            return seq.tolist().index(vocab[self.eos_token]) + 1
        except ValueError:
            return len(seq)


    # modified by Zhaopeng Tu, 2016-05-04
    # def _idx_to_word(self, seq, ivocab):
    def _idx_to_word(self, seq, ivocab, coverage=None):
        if coverage is None:
            return " ".join([ivocab.get(idx, self.unk_token) for idx in seq])
        else:
            output = []
            for i, [idx, ratio] in enumerate(zip(seq, coverage)):
                output.append('%s/%.2f' % (ivocab.get(idx, self.unk_token), ratio))
            return " ".join(output)


class BleuValidator(object):

    def __init__(self, search_model, test_src=None, test_ref=None, **kwards):

        self.search_model = search_model
        self.unk_token = kwards.pop('unk_token')
        self.unk_id = kwards.pop('unk_id')
        self.eos_token = kwards.pop('eos_token')
        self.vocab_src = kwards.pop('vocab_src')
        self.vocab_trg = kwards.pop('vocab_trg')
        self.bleu_script = kwards.pop('bleu_script')
        self.res_to_sgm = kwards.pop('res_to_sgm')
        self.test_src = test_src
        self.test_ref = test_ref
 
        # added by Zhaopeng Tu, 2017-05-22
        self.with_subword = kwards.pop('with_subword')
        self.with_europe_eval = kwards.pop('with_europe_eval')
        # added by Zhaopeng Tu, 2016-07-29
        self.output_kbest = kwards.pop('output_kbest')
      
        # added by Zhaopeng Tu, 2016-05-04
        self.with_coverage = kwards.pop('with_coverage')
        self.coverage_type = kwards.pop('coverage_type')
        
        # added by Zhaopeng Tu, 2016-07-19
        self.with_reconstruction = kwards.pop('with_reconstruction')

        # added by Zhaopeng Tu, 2016-07-21
        # replace unk
        self.replace_unk = kwards.pop('replace_unk')
        if self.replace_unk:
            self.unk_dict = self.read_dict(kwards.pop('unk_dict'))

        if test_src is None or test_ref is None:
            self.test_src = kwards.pop('valid_src')
            self.test_ref = kwards.pop('valid_trg')

        self.dict_src, self.idict_src = self._get_dict(self.vocab_src)
        self.dict_trg, self.idict_trg = self._get_dict(self.vocab_trg)

    def read_dict(self, dict_file):
        word_dict = {}
        fin = open(dict_file)
        while 1:
            try:
                line = fin.next().strip()
            except StopIteration:
                break

            src, tgt, prob = line.split()
            prob = float(prob)
            if (src not in word_dict) or prob > word_dict[src][1]:
                word_dict[src] = (tgt, prob)

        for key in word_dict:
            word_dict[key] = word_dict[key][0]

        return word_dict


    def replace_translation_unk(self, source_words_dict_trans, output, alignment):
        tran_words = self._idx_to_word(output, self.idict_trg).split()
        aligned_source_words_dict_trans = [source_words_dict_trans[idx] for idx in numpy.argmax(alignment, axis=0)]
        new_tran_words = []
        for i in xrange(len(tran_words)):
            if tran_words[i] != self.unk_token:
                new_tran_words.append(tran_words[i])
            else:
                # replace unk token
                new_tran_words.append(aligned_source_words_dict_trans[i])

        return " ".join(new_tran_words)


    def apply(self, data_stream, in_file, out_file, verbose=False):
        logger.info("Begin decoding ...")
        fin = open(in_file)
        fout = open(out_file, 'w')
        if self.output_kbest:
            fout_kbest = open(out_file+'.kbest', 'w')

        val_start_time = time.time()
        i = 0
        for sent in data_stream.get_epoch_iterator():
            if self.replace_unk:
                source_sent = fin.next().strip()
                source_words = source_sent.split() + [self.eos_token]
                source_words_dict_trans = [self.unk_dict.get(w, w) for w in source_words]

            i += 1
            # modified by Zhaopeng Tu, 2016-05-04
            # outputs, scores = self.search_model.apply(numpy.array(sent).T)
            results = self.search_model.apply(numpy.array(sent).T)
            outputs, scores, alignments = results[:3]
            index = 3
            if self.with_coverage:
                coverages = results[index]
                index += 1
                if self.coverage_type is 'linguistic':
                    fertilities = results[index]
                    index += 1

            if self.with_reconstruction:
                reconstruction_scores = results[index]
                inverse_alignments = results[index+1]
                index += 2

            sidx = numpy.argmin(scores)

            # added by Zhaopeng Tu, 2016-07-21
            if self.replace_unk:
                res = self.replace_translation_unk(source_words_dict_trans, outputs[sidx][:-1], numpy.array(alignments[sidx]).transpose())
            else:
                res = self._idx_to_word(outputs[sidx][:-1], self.idict_trg)

            if res.strip() == '':
                # <UNK> bug
                # res = self.unk_token
                res = 'UNK'

            fout.write(res + '\n')

            if self.output_kbest:
                for idx in xrange(len(outputs)):
                    kbest_score = [str(scores[idx])]
                    aligns = [str(numpy.array(alignments[idx]).transpose().tolist())]
                    if self.with_reconstruction:
                        kbest_score.extend([str(scores[idx]-reconstruction_scores[idx]), str(reconstruction_scores[idx])])
                        aligns.append(str(numpy.array(inverse_alignments[idx]).tolist()))
                    
                    if self.replace_unk:
                        alignment = numpy.array(alignments[idx]).transpose()
                        res = self.replace_translation_unk(source_words_dict_trans, outputs[idx][:-1], alignment)
                    else:
                        res = self._idx_to_word(outputs[idx][:-1], self.idict_trg)
                
                    print >> fout_kbest, '%d ||| %s ||| %s ||| %s' % (i, ' ||| '.join(kbest_score), res , ' ||| '.join(aligns))


            # added by Zhaopeng Tu, 2016-05-04
            if verbose:
                print 'Parsed Input:', self._idx_to_word(sent[0], self.idict_src)
                # output alignment and coverage information
                print 'Translation:', res
                print 'Score:', scores[sidx]
                print 'Aligns:'
                print numpy.array(alignments[sidx]).transpose().tolist()
                if self.with_coverage:
                    coverage = coverages[sidx]
                    # sent is a batch that contains only one sentence
                    sentence = [self.idict_src[idx] for idx in sent[0]]
                    print 'Coverage:',
                    for k in xrange(len(sentence)):
                        print '%s/%.2f' % (sentence[k], coverage[k]),
                    print ''
                    if self.coverage_type is 'linguistic':
                        print 'Fertility:',
                        for k in xrange(len(sentence)):
                            print '%s/%.2f' % (sentence[k], fertilities[k]),
                        print ''

                if self.with_reconstruction:
                    print 'Reconstruction Score:', reconstruction_scores[sidx]
                    print 'Inverse Aligns:'
                    print numpy.array(inverse_alignments[sidx]).tolist()

            if i % 100 == 0:
                logger.info("Translated {} lines of valid/test set ...".format(i))

        fin.close()
        fout.close()
        if self.output_kbest:
            fout_kbest.close()

        logger.info("Decoding took {} minutes".format(float(time.time() - val_start_time) / 60.))

        logger.info("Evaluate ...")

        # added by Zhaopeng Tu, 2017-05-22
        # for subword
        if self.with_subword:
            logger.info('transforming subword to word')
            os.system("mv %s %s.subword; sed -r 's/(@@ )|(@@ ?$)//g' %s.subword > %s" % (out_file, out_file, out_file, out_file))

        if self.with_europe_eval:
            bleu_cmd = ['perl', self.bleu_script, \
                            self.test_ref, \
                            '<', out_file, \
                            '>', out_file+'.eval']
            logger.info('compute bleu score')
            subprocess.check_call(" ".join(bleu_cmd), shell=True)

            fin = open(out_file+'.eval', 'rU')
            out = re.search('BLEU = [-.0-9]+', fin.readlines()[0])
            fin.close()

            bleu_score = float(out.group()[7:])
            logger.info("Done")
        else:
            cmd_res_to_sgm = ['python', self.res_to_sgm, out_file, self.test_src+'.sgm', out_file+'.sgm']
            cmd_bleu_cmd = ['perl', self.bleu_script, \
                            '-r', self.test_ref+'.sgm', \
                            '-s', self.test_src+'.sgm', \
                            '-t', out_file+'.sgm', \
                            '>', out_file+'.eval']

            logger.info('covert result to sgm')
            subprocess.check_call(" ".join(cmd_res_to_sgm), shell=True)
            logger.info('compute bleu score')
            subprocess.check_call(" ".join(cmd_bleu_cmd), shell=True)

            fin = open(out_file+'.eval', 'rU')
            out = re.search('BLEU score = [-.0-9]+', fin.readlines()[7])
            fin.close()

            bleu_score = float(out.group()[13:])
            logger.info("Done")

        return bleu_score


    def _get_dict(self, vocab_file):

        if os.path.isfile(vocab_file):
            ddict = pkl.load(open(vocab_file, 'rb'))
        else:
            logger.error("file [{}] do not exist".format(vocab_file))

        iddict = dict()
        for kk, vv in ddict.iteritems():
            iddict[vv] = kk

        iddict[0] = self.eos_token
        # <UNK> bug, it will be tokenized into '<', 'UNK', '>' by mteval.perl
        iddict[1] = 'UNK'

        return ddict, iddict


    def _idx_to_word(self, seq, ivocab):
        return " ".join([ivocab.get(idx, self.unk_token) for idx in seq])


if __name__=='__main__':
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto", default="get_config_search_coverage",
                        help="Prototype config to use for config")
    # added by Zhaopeng Tu, 2016-05-12
    parser.add_argument("--state", help="State to use")
    # added by Zhaopeng Tu, 2016-05-27
    parser.add_argument("--model", help="Model to use")
    # added by Zhaopeng Tu, 2016-07-20
    parser.add_argument("--beam", type=int, default=10, help="Beam size")
    # added by Zhaopeng Tu, 2016-11-08
    parser.add_argument("--length_penalty_factor", type=float, default=0., help="Weight factor of length penalty")
    parser.add_argument("--coverage_penalty_factor", type=float, default=0., help="Weight factor of coverage penalty")
    parser.add_argument('source', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('trans', type=str)
    args = parser.parse_args()

    configuration = getattr(configurations, args.proto)()
    # added by Zhaopeng Tu, 2016-05-12
    if args.state:
        configuration.update(eval(open(args.state).read()))
    logger.info("\nModel options:\n{}".format(pprint.pformat(configuration)))

    rng = numpy.random.RandomState(1234)

    enc_dec = EncoderDecoder(rng, **configuration)
    enc_dec.build_sampler()

    # added by Zhaopeng Tu, 2016-05-27
    # options to use other trained models
    if args.model:
        enc_dec.load(path=args.model)
    else:
        enc_dec.load(path=configuration['saveto_best'])

    # added by Zhaopeng Tu, 2016-11-08
    if args.length_penalty_factor > 0.:
        configuration['length_penalty_factor'] = args.length_penalty_factor
    if args.coverage_penalty_factor > 0.:
        configuration['coverage_penalty_factor'] = args.coverage_penalty_factor

    # added by Zhaopeng Tu, 2016-07-20
    beam_size = configuration['beam_size']
    if args.beam:
        beam_size = args.beam

    test_search = BeamSearch(enc_dec=enc_dec,
                             configuration=configuration, 
                             beam_size=beam_size,
                             maxlen=3*configuration['seq_len_src'], stochastic=False)
    bleuvalidator = BleuValidator(search_model=test_search, test_src=args.source, test_ref=args.target, **configuration)

    # test data
    ts = get_devtest_stream(data_type='test', input_file=args.source, **configuration)
    test_bleu = bleuvalidator.apply(ts, args.source, args.trans, True)

    logger.info('test bleu %.4f' %test_bleu)

