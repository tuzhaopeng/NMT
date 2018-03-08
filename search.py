# models: BeamSearch
import numpy
import copy


class BeamSearch(object):
    def __init__(self, enc_dec, configuration, beam_size=1, maxlen=50, stochastic=True):

        self.enc_dec = enc_dec
        # if sampling, beam_size = 1
        self.beam_size = beam_size
        # max length of output sentence
        self.maxlen = maxlen
        # stochastic == True stands for sampling
        self.stochastic = stochastic
        
        # added by Zhaopeng Tu, 2016-11-08
        self.length_penalty_factor = configuration['length_penalty_factor']
        self.with_length_penalty = True if self.length_penalty_factor > 0. else False
        self.coverage_penalty_factor = configuration['coverage_penalty_factor']
        self.with_coverage_penalty = True if self.coverage_penalty_factor > 0. else False

        self.with_decoding_pruning = configuration['with_decoding_pruning']
        self.decoding_pruning_beam = configuration['decoding_pruning_beam']

        # added by Zhaopeng Tu, 2016-06-09
        self.with_attention = configuration['with_attention']
       
        # added by Zhaopeng Tu, 2016-05-04
        self.with_coverage = configuration['with_coverage']
        self.coverage_dim = configuration['coverage_dim']
        self.coverage_type = configuration['coverage_type']
        self.max_fertility = configuration['max_fertility']
        
        # added by Zhaopeng Tu, 2016-07-19
        self.with_reconstruction = configuration['with_reconstruction']
        self.reconstruction_weight = configuration['reconstruction_weight']

        if self.beam_size > 1:
            assert not self.stochastic, 'Beam search does not support stochastic sampling'


    def cal_penalty_score(self, log_score, length, alignment):
        score = log_score
        if self.with_length_penalty:
            length_penalty = pow(5.+length, self.length_penalty_factor) / pow(5.+1, self.length_penalty_factor)
            score = score / length_penalty
        if self.with_coverage_penalty:
            coverage = numpy.minimum(numpy.sum(alignment, axis=0), 1.)
            coverage_penalty = self.coverage_penalty_factor * numpy.sum(numpy.log(coverage))
            score += coverage_penalty

        return score



    def apply(self, input):
        sample = []
        sample_score = []
        if self.stochastic:
            sample_score = 0
        # added by Zhaopeng Tu, 2016-05-03
        sample_alignment = []
        if self.with_coverage:
            sample_coverage = []
        # added by Zhaopeng Tu, 2016-07-18
        # for reconstruction
        if self.with_reconstruction:
            sample_states = []

        # get initial state of decoder rnn and encoder context
        ret = self.enc_dec.compile_init_and_context(input)
        next_state, c = ret[0], ret[1]
 
        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = numpy.zeros(live_k).astype('float32')
        hyp_states = [[]] * live_k
        # added by Zhaopeng Tu, 2016-05-02, coverage model 
        hyp_alignments = [[]] * live_k
        # note that batch size is the second dimension coverage and will be used in the later decoding, thus we need a structure different from the above ones 
        if self.with_coverage:
            hyp_coverages  = numpy.zeros((c.shape[0], live_k, self.coverage_dim), dtype='float32')
            if self.coverage_type is 'linguistic':
                # note the return result is a list even when it contains only one element
                fertility = self.enc_dec.compile_fertility(c)[0]

        # bos indicator
        next_w = -1 * numpy.ones((1,)).astype('int64')

        for i in range(self.maxlen):
            ctx = numpy.tile(c, [live_k, 1])
 
            inps = [next_w, next_state, ctx]
           
            # added by Zhaopeng Tu, 2016-05-03
            if self.with_coverage:
                inps.append(hyp_coverages)
                if self.coverage_type is 'linguistic':
                    inps.append(fertility)

            ret = self.enc_dec.compile_next_state_and_probs(*inps)
            next_p, next_state, next_w, alignment = ret[:4]
            idx = 4

            # added by Zhaopeng Tu, 2016-05-03
            # update the coverage after attention operation
            if self.with_coverage:
                coverages = ret[idx]
                idx += 1
                
            if self.stochastic:
                nw = next_w[0]
                sample.append(nw)
                if self.with_reconstruction:
                    sample_states.append(next_state[0])
                sample_score -= numpy.log(next_p[0, nw])
                # added by Zhaopeng Tu, 2016-05-12
                if self.with_coverage:
                    hyp_coverages = coverages
                # 0 for EOS
                if nw == 0:
                    break
            else:
                cand_scores = hyp_scores[:, None] - numpy.log(next_p)
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:self.beam_size-dead_k]

                voc_size = next_p.shape[1]
                trans_indices = ranks_flat / voc_size
                word_indices = ranks_flat % voc_size
                costs = cand_flat[ranks_flat]

                new_hyp_samples = []
                new_hyp_scores = numpy.zeros(self.beam_size-dead_k).astype('float32')
                new_hyp_states = []
                # added by Zhaopeng Tu, 2016-05-03, coverage model
                new_hyp_alignments = []
                if self.with_coverage:
                    new_hyp_coverages = numpy.zeros((c.shape[0], self.beam_size-dead_k, self.coverage_dim), dtype='float32')

                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[ti]+[wi])
                    new_hyp_scores[idx] = costs[idx]
                    new_hyp_states.append(hyp_states[ti]+[next_state[ti]])

                    # added by Zhaopeng Tu, 2016-05-03, coverage model
                    new_hyp_alignments.append(hyp_alignments[ti]+[alignment[:,ti]])
                    if self.with_coverage:
                        new_hyp_coverages[:,idx,:] = coverages[:, ti, :] 

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []
                # added by Zhaopeng Tu, 2016-05-03
                hyp_alignments = []
                if self.with_coverage:
                    indices = []
                
                for idx in range(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == 0:
                        sample.append(new_hyp_samples[idx])
                        # added by Zhaopeng Tu, 2016-11-08
                        # sample_score.append(new_hyp_scores[idx])
                        # calculate normalized score
                        sample_score.append(self.cal_penalty_score(new_hyp_scores[idx], len(new_hyp_samples[idx]), new_hyp_alignments[idx]) )
                        
                        # added by Zhaopeng Tu, 2016-07-18
                        # for reconstruction
                        if self.with_reconstruction:
                            sample_states.append(new_hyp_states[idx])

                        # added by Zhaopeng Tu, 2016-05-03
                        sample_alignment.append(new_hyp_alignments[idx])
                        if self.with_coverage:
                            # for neural coverage, we use the mean value of the vector
                            sample_coverage.append(new_hyp_coverages[:,idx,:].mean(1))

                        dead_k += 1
                    else:
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        hyp_states.append(new_hyp_states[idx])

                        hyp_alignments.append(new_hyp_alignments[idx])
                        if self.with_coverage:
                            indices.append(idx)

                        new_live_k += 1

                hyp_scores = numpy.array(hyp_scores)
                live_k = new_live_k
                
                # added by Zhaopeng Tu, 2016-05-04
                if self.with_coverage:
                    # note now liv_k has changed
                    hyp_coverages  = numpy.zeros((c.shape[0], live_k, self.coverage_dim), dtype='float32')
                    for idx in xrange(live_k):
                        hyp_coverages[:,idx,:] = new_hyp_coverages[:, indices[idx], :]

                if live_k < 1 or dead_k >= self.beam_size:
                    break

                next_w = numpy.array([w[-1] for w in hyp_samples])
                next_state = numpy.array([s[-1] for s in hyp_states])

        if not self.stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in range(live_k):
                    sample.append(hyp_samples[idx])
                    # added by Zhaopeng Tu, 2016-11-08
                    # sample_score.append(hyp_scores[idx])
                    # calculate normalized score
                    sample_score.append( self.cal_penalty_score(hyp_scores[idx], len(hyp_samples[idx]), hyp_alignments[idx]) )

                    # added by Zhaopeng Tu, 2016-05-04
                    sample_alignment.append(hyp_alignments[idx])
                    if self.with_coverage:
                        sample_coverage.append(hyp_coverages[:,idx,:].mean(1))
                    # added by Zhaopeng Tu, 2016-07-18
                    if self.with_reconstruction:
                        sample_states.append(hyp_states[idx])

        # added by Zhaopeng Tu, 2016-05-12
        else:
            if self.with_coverage:
                sample_coverage = hyp_coverages[:,0,:].mean(1)

        # added by Zhaopeng Tu, 2016-07-18
        # for reconstruction
        if self.with_reconstruction:
            # build inverce_c and mask
            if self.stochastic:
                sample_states = [sample_states]
            sample_num = len(sample_states)

            inverse_sample_score = numpy.zeros(sample_num).astype('float32')
            # added by Zhaopeng Tu, 2016-05-03
            inverse_sample_alignment = [[] for i in  xrange(sample_num)]

            my = max([len(s) for s in sample_states])
            inverse_c = numpy.zeros((my, sample_num, sample_states[0][0].shape[0]), dtype='float32')
            mask = numpy.zeros((my, sample_num), dtype='float32')
            for idx in range(sample_num):
                inverse_c[:len(sample_states[idx]), idx, :] = sample_states[idx]
                mask[:len(sample_states[idx]), idx] = 1.

            # get initial state of decoder rnn and encoder context
            inverse_ret = self.enc_dec.compile_inverse_init_and_context(inverse_c)
            inverse_next_state = inverse_ret[0]

            to_reconstruct_input = input[:,0]
            for i in range(len(to_reconstruct_input)):
                # whether input contains eos?
                inverse_next_w = numpy.array([to_reconstruct_input[i-1]]*sample_num) if i>0 else -1 * numpy.ones((sample_num,)).astype('int64')
                
                inps = [inverse_next_w, mask, inverse_next_state, inverse_c]

                ret = self.enc_dec.compile_inverse_next_state_and_probs(*inps)
                inverse_next_p, inverse_next_state, inverse_next_w, inverse_alignment = ret[:4]

                # compute reconstruction error
                inverse_sample_score -= numpy.log(inverse_next_p[:, to_reconstruct_input[i]])

                # for each sample
                for idx in range(sample_num):
                    inverse_sample_alignment[idx].append(inverse_alignment[:len(sample_states[idx]), idx])

            # combine sample_score and reconstructed_score
            sample_score += inverse_sample_score * self.reconstruction_weight


        results = [sample, sample_score, sample_alignment]
        if self.with_coverage:
            results.append(sample_coverage)
            if self.coverage_type is 'linguistic':
                results.append(fertility[:,0])

        if self.with_reconstruction:
            results.append(inverse_sample_score)
            results.append(inverse_sample_alignment)

        return results



# added by Zhaopeng Tu, 2016-12-22
# Force decoding a given sentence pair (source, target)
class ForceDecoding(object):
    def __init__(self, enc_dec, configuration):
        self.enc_dec = enc_dec
       
        self.with_coverage = configuration['with_coverage']
        self.coverage_dim = configuration['coverage_dim']
        self.coverage_type = configuration['coverage_type']
        self.max_fertility = configuration['max_fertility']
        
        # to judege whether to return decoder_states
        self.with_reconstruction = configuration['with_reconstruction']


    def apply(self, source, target):
        cost = []
        alignment = []

        if self.with_reconstruction:
            decoder_states = numpy.zeros((target.shape[0], 1, next_state.shape[1]), dtype='float32')

        ### Force decode sentence pair (source, target)
        # get initial state of decoder rnn and encoder context
        ret = self.enc_dec.compile_init_and_context(source)
        next_state, ctx = ret[0], ret[1]

        if self.with_coverage:
            coverage  = numpy.zeros((ctx.shape[0], 1, self.coverage_dim), dtype='float32')
            if self.coverage_type is 'linguistic':
                # note the return result is a list even when it contains only one element
                fertility = self.enc_dec.compile_fertility(ctx)[0]

        for i in range(len(target)):
            next_w = numpy.array(target[i-1]) if i > 0 else -1 * numpy.ones((1,)).astype('int64')
            inps = [next_w, next_state, ctx]
                
            if self.with_coverage:
                inps.append(coverage)
                if self.coverage_type is 'linguistic':
                    inps.append(fertility)

            ret = self.enc_dec.compile_next_state_and_probs(*inps)
            next_p, next_state, next_w, align = ret[:4]
            idx = 4

            # update the coverage after attention operation
            if self.with_coverage:
                coverage = ret[idx]
                idx += 1
            
            # added by Zhaopeng Tu, 2017-06-12
            # cost -= numpy.log(next_p[0, target[i]])
            cost.append(next_p[0, target[i]][0])
            alignment.append(align[:,0])
            if self.with_reconstruction:
                decoder_states[i,0,:] = next_state[0,:]

        results = [cost, alignment]
        if self.with_coverage:
            coverage = coverage[:,0,:].mean(1)
            results.append(coverage)
            if self.coverage_type is 'linguistic':
                results.append(fertility[:,0])

        # the last one is easy to be indexed, we leave it for reconstruction
        if self.with_reconstruction:
            results.append(decoder_states)

        return results


# added by Zhaopeng Tu, 2016-06-11
# for forced alignment
class Align(object):
    def __init__(self, enc_dec, configuration):
        self.enc_dec = enc_dec

        self.with_reconstruction = configuration['with_reconstruction']

        self.force_decoder = ForceDecoding(enc_dec, configuration)


    def apply(self, source, target):
        results = self.force_decoder.apply(source, target)

        # added by Zhaopeng Tu, 2016-07-16
        if self.with_reconstruction:
            inverse_alignment = []

            # fetch decoder states from results
            decoder_states = results[-1]
            results = results[:-1]

            ret = self.enc_dec.compile_inverse_init_and_context(decoder_states)
            inverse_next_state = ret[0]

            mask = numpy.ones((decoder_states.shape[0], 1), dtype='float32')

            for i in range(len(source)):
                inverse_next_w = numpy.array(source[i-1]) if i > 0 else -1 * numpy.ones((1,)).astype('int64')
                inps = [inverse_next_w, mask, inverse_next_state, decoder_states]

                ret = self.enc_dec.compile_inverse_next_state_and_probs(*inps)
                inverse_next_p, inverse_next_state, inverse_next_w, inverse_align = ret[:4]

                inverse_alignment.append(inverse_align[:,0])

            results.append(inverse_alignment)

        return results

