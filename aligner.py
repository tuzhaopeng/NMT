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
import copy
import configurations
from search import Align
from nmt import EncoderDecoder
from stream import get_stream

logger = logging.getLogger(__name__)


class Aligner(object):

    def __init__(self, align_model, test_src, test_trg, **kwards):
        self.align_model = align_model
        self.test_src = test_src
        self.test_trg = test_trg

        self.unk_token = kwards.pop('unk_token')
        self.eos_token = kwards.pop('eos_token')
        self.vocab_src = kwards.pop('vocab_src')
        self.vocab_trg = kwards.pop('vocab_trg')
       
        # added by Zhaopeng Tu, 2016-06-09
        self.with_attention = kwards.pop('with_attention')

        # added by Zhaopeng Tu, 2016-05-04
        self.with_coverage = kwards.pop('with_coverage')
        self.coverage_type = kwards.pop('coverage_type')
        
        # added by Zhaopeng Tu, 2016-07-19
        self.with_reconstruction = kwards.pop('with_reconstruction')

        self.dict_src, self.idict_src = self._get_dict(self.vocab_src)
        self.dict_trg, self.idict_trg = self._get_dict(self.vocab_trg)


    def apply(self, output, verbose=False):
        fout = open(output, 'w')
        if self.with_reconstruction:
            fout_inverse = open(output+'.inverse', 'w')
     
            # alignment symmetrization
            fout_forward = open(output+'.forward', 'w')
            fout_backward = open(output+'.backward', 'w')
            fout_intersection = open(output+'.intersection', 'w')
            fout_union = open(output+'.union', 'w')
            fout_grow = open(output+'.grow', 'w')
            fout_diag = open(output+'.diag', 'w')
            fout_gd = open(output+'.gd', 'w')
            fout_gdf = open(output+'.gdf', 'w')
            fout_gdfa = open(output+'.gdfa', 'w')
            
       
        while 1:
            try:
                source = self.test_src.get_data()
                target = self.test_trg.get_data()
            except:
                break

            results = self.align_model.apply(numpy.array(source).T, numpy.array(target).T)
            cost = results[0]
            alignment = numpy.array(results[1]).transpose()
            idx = 2
            if self.with_coverage:
                coverage = results[idx]
                idx += 1
                if self.coverage_type is 'linguistic':
                    fertility = results[idx]
                    idx += 1
            
            print >> fout, alignment.tolist()
                
            if self.with_reconstruction:
                inverse_alignment = numpy.array(results[idx])
                idx += 1
                
                print >> fout_inverse, inverse_alignment.tolist()

                forward, backward, intersection, union, grow_align, diag_align, gd_align, gdf_align, gdfa_align = self.grow_diag_final_and(alignment, inverse_alignment)
                print >> fout_forward, numpy.array(forward).tolist()
                print >> fout_backward, numpy.array(backward).tolist()
                print >> fout_intersection, numpy.array(intersection).tolist()
                print >> fout_union, numpy.array(union).tolist()
                print >> fout_grow, numpy.array(grow_align).tolist()
                print >> fout_diag, numpy.array(diag_align).tolist()
                print >> fout_gd, numpy.array(gd_align).tolist()
                print >> fout_gdf, numpy.array(gdf_align).tolist()
                print >> fout_gdfa, numpy.array(gdfa_align).tolist()


            if verbose:
                source = source[0]
                target = target[0]
                print "Parsed Input:", self._idx_to_word(source, self.idict_src)
                print "Translation:", self._idx_to_word(target, self.idict_trg)
                print "Score: %.4f" % (-1.*numpy.log(cost).sum())
                print cost
                print "Aligns:"
                print alignment.tolist()
                # added by Zhaopeng Tu, 2016-05-04
                if self.with_coverage:
                    print "Coverage:", self._idx_to_word(source, self.idict_src, coverage)
                    if self.coverage_type is 'linguistic':
                        print "Fertility:", self._idx_to_word(source, self.idict_src, fertility)

                # added by Zhaopeng Tu, 2016-07-19
                if self.with_reconstruction:
                    print "Inverse Aligns:"
                    print inverse_alignment.tolist()
                    if self.with_coverage:
                        print "Inverse Coverage:", self._idx_to_word(target, self.idict_trg, inverse_coverage)
                        if self.coverage_type is 'linguistic':
                            print "Inverse Fertility:", self._idx_to_word(target, self.idict_trg, inverse_fertility)


    def grow_diag_final_and(self, alignment, inverse_alignment):
        # get hard align
        s2t = alignment >= alignment.max(axis=0, keepdims=True)

        # get hard align
        t2s = inverse_alignment >= inverse_alignment.max(axis=1, keepdims=True)

        intersection = s2t * t2s
        union = s2t + t2s

        grow_alignment = copy.copy(intersection)
        self.grow(grow_alignment, union)

        diag_alignment = copy.copy(intersection)
        self.diag(diag_alignment, union)

        gd_alignment = copy.copy(intersection)
        self.grow_diag(gd_alignment, union)

        gdf_alignment = copy.copy(gd_alignment)
        self.final(gdf_alignment, s2t)
        self.final(gdf_alignment, t2s)

        gdfa_alignment = copy.copy(gd_alignment)
        self.final_and(gdfa_alignment, s2t)
        self.final_and(gdfa_alignment, t2s)

        return s2t, t2s, intersection, union, grow_alignment, diag_alignment, gd_alignment, gdf_alignment, gdfa_alignment

    def grow_diag(self, refined_alignment, union):
        neighbours = ((-1,0),(0,-1),(1,0),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))
        for i in xrange(refined_alignment.shape[0]):
            for j in xrange(refined_alignment.shape[1]):
                if not refined_alignment[i,j]:
                    continue

                cell = (i,j)
                for n in neighbours:
                    neighbour = (cell[0]+n[0], cell[1]+n[1])
                    if neighbour[0]<0 or neighbour[0]>=refined_alignment.shape[0] or neighbour[1]<0 or neighbour[1]>=refined_alignment.shape[1]:
                        continue

                    if (sum(refined_alignment[neighbour[0],:]) == 0 or sum(refined_alignment[:, neighbour[1]]) == 0) and union[neighbour]:
                        refined_alignment[neighbour] = True


    def grow(self, refined_alignment, union):
        neighbours = ((-1,0),(0,-1),(1,0),(0,1))
        for i in xrange(refined_alignment.shape[0]):
            for j in xrange(refined_alignment.shape[1]):
                if not refined_alignment[i,j]:
                    continue

                cell = (i,j)
                for n in neighbours:
                    neighbour = (cell[0]+n[0], cell[1]+n[1])
                    if neighbour[0]<0 or neighbour[0]>=refined_alignment.shape[0] or neighbour[1]<0 or neighbour[1]>=refined_alignment.shape[1]:
                        continue

                    if (sum(refined_alignment[neighbour[0],:]) == 0 or sum(refined_alignment[:, neighbour[1]]) == 0) and union[neighbour]:
                        refined_alignment[neighbour] = True


    def diag(self, refined_alignment, union):
        neighbours = ((-1,-1),(-1,1),(1,-1),(1,1))
        for i in xrange(refined_alignment.shape[0]):
            for j in xrange(refined_alignment.shape[1]):
                if not refined_alignment[i,j]:
                    continue

                cell = (i,j)
                for n in neighbours:
                    neighbour = (cell[0]+n[0], cell[1]+n[1])
                    if neighbour[0]<0 or neighbour[0]>=refined_alignment.shape[0] or neighbour[1]<0 or neighbour[1]>=refined_alignment.shape[1]:
                        continue

                    if (sum(refined_alignment[neighbour[0],:]) == 0 or sum(refined_alignment[:, neighbour[1]]) == 0) and union[neighbour]:
                        refined_alignment[neighbour] = True


    def final_and(self, refined_alignment, direction_alignment):
        for i in xrange(refined_alignment.shape[0]):
            for j in xrange(refined_alignment.shape[1]):
                if (sum(refined_alignment[i,:]) == 0 and sum(refined_alignment[:,j]) == 0) and direction_alignment[i,j]:
                    refined_alignment[i,j] = True
 
    def final(self, refined_alignment, direction_alignment):
        for i in xrange(refined_alignment.shape[0]):
            for j in xrange(refined_alignment.shape[1]):
                if (sum(refined_alignment[i,:]) == 0 or sum(refined_alignment[:,j]) == 0) and direction_alignment[i,j]:
                    refined_alignment[i,j] = True
 

    def _get_dict(self, vocab_file):

        if os.path.isfile(vocab_file):
            ddict = pkl.load(open(vocab_file, 'rb'))
        else:
            logger.error("file [{}] do not exist".format(vocab_file))

        iddict = dict()
        for kk, vv in ddict.iteritems():
            iddict[vv] = kk

        iddict[0] = self.eos_token

        return ddict, iddict

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


if __name__=='__main__':
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto", default="get_config_search_coverage",
                        help="Prototype config to use for config")
    # added by Zhaopeng Tu, 2016-05-12
    parser.add_argument("--state", help="State to use")
    # added by Zhaopeng Tu, 2016-05-27
    parser.add_argument("--model", help="Model to use")
    parser.add_argument('source', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('alignment', type=str)
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

    test_align = Align(enc_dec, configuration)
    test_src = get_stream(args.source, configuration['vocab_src'], **configuration)
    test_trg = get_stream(args.target, configuration['vocab_trg'], **configuration)
    aligner = Aligner(align_model=test_align, test_src=test_src, test_trg=test_trg, **configuration)

    aligner.apply(args.alignment, True)
