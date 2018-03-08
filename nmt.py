# rnn encoder-decoder for machine translation
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import logging
import os
from models import LookupTable, LogisticRegression, BidirectionalEncoder, Decoder
from utils import Dropout
from algorithm import adadelta, adam, grad_clip
from theano.ifelse import ifelse
# added by Zhaopeng Tu, 2016-07-30
# for debugging NaN
from theano.compile.nanguardmode import NanGuardMode

logger = logging.getLogger(__name__)


class EncoderDecoder(object):

    def __init__(self, rng, **kwargs):
        self.n_in_src = kwargs.get('nembed_src')
        self.n_in_trg = kwargs.get('nembed_trg')
        self.n_hids_src = kwargs.get('nhids_src')
        self.n_hids_trg = kwargs.get('nhids_trg')
        self.src_vocab_size = kwargs.get('src_vocab_size')
        self.trg_vocab_size = kwargs.get('trg_vocab_size')
        self.method = kwargs.get('method')
        self.dropout = kwargs.get('dropout')
        self.maxout_part = kwargs.get('maxout_part')
        self.path = kwargs.get('saveto')
        self.clip_c = kwargs.get('clip_c')
        self.rng = rng
        self.trng = RandomStreams(rng.randint(1e5))

        # added by Zhaopeng Tu, 2016-04-29
        self.with_coverage = kwargs.get('with_coverage')
        self.coverage_dim = kwargs.get('coverage_dim')
        self.coverage_type = kwargs.get('coverage_type')
        self.max_fertility = kwargs.get('max_fertility')
        if self.coverage_type is 'linguistic':
            # make sure the dimension of linguistic coverage is always 1
            self.coverage_dim = 1
        
        # added by Zhaopeng Tu, 2016-05-30
        self.with_context_gate = kwargs.get('with_context_gate')

        # added by Zhaopeng Tu, 2017-11-29
        self.with_layernorm = kwargs.get('with_layernorm', False)

        self.params = []
        self.layers = []

        self.table_src = LookupTable(self.rng, self.src_vocab_size, self.n_in_src, name='table_src')
        self.layers.append(self.table_src)

        self.encoder = BidirectionalEncoder(self.rng, self.n_in_src, self.n_hids_src, self.table_src, name='birnn_encoder')
        self.layers.append(self.encoder)

        self.table_trg = LookupTable(self.rng, self.trg_vocab_size, self.n_in_trg, name='table_trg')
        self.layers.append(self.table_trg)

        self.decoder = Decoder(self.rng, self.n_in_trg, self.n_hids_trg, 2*self.n_hids_src, \
                               maxout_part=self.maxout_part, name='rnn_decoder', \
                               # added by Zhaopeng Tu, 2016-04-29
                               with_coverage=self.with_coverage, coverage_dim=self.coverage_dim, coverage_type=self.coverage_type, max_fertility=self.max_fertility, \
                               # added by Zhaopeng Tu, 2016-05-30
                               with_context_gate=self.with_context_gate, \
                               with_layernorm=self.with_layernorm)
        self.layers.append(self.decoder)

        self.logistic_layer = LogisticRegression(self.rng, self.n_in_trg, self.trg_vocab_size)
        self.layers.append(self.logistic_layer)

        # added by Zhaopeng Tu, 2016-07-12
        # for reconstruction
        self.with_reconstruction = kwargs.get('with_reconstruction')
        if self.with_reconstruction:
            # added by Zhaopeng Tu, 2016-07-27
            self.reconstruction_weight = kwargs.get('reconstruction_weight')
            # note the source and target sides are reversed
            self.inverse_decoder = Decoder(self.rng, self.n_in_src, 2*self.n_hids_src, self.n_hids_trg, \
                                   maxout_part=self.maxout_part, name='rnn_inverse_decoder', \
                                   with_layernorm=self.with_layernorm)
            self.layers.append(self.inverse_decoder)
            
            self.srng = RandomStreams(rng.randint(1e5))
            self.inverse_logistic_layer = LogisticRegression(self.rng, self.n_in_src, self.src_vocab_size, name='inverse_LR')
            self.layers.append(self.inverse_logistic_layer)

        for layer in self.layers:
            self.params.extend(layer.params)


    def build_trainer(self, src, src_mask, trg, trg_mask):
        annotations = self.encoder.apply(src, src_mask)
        # init_context = annotations[0, :, -self.n_hids_src:]
        # modification #1
        # mean pooling
        init_context = (annotations * src_mask[:, :, None]).sum(0) / src_mask.sum(0)[:, None]

        trg_emb = self.table_trg.apply(trg)
        trg_emb_shifted = T.zeros_like(trg_emb)
        trg_emb_shifted = T.set_subtensor(trg_emb_shifted[1:], trg_emb[:-1])
        results = self.decoder.run_pipeline(state_below=trg_emb_shifted,
                                            mask_below=trg_mask,
                                            init_context=init_context,
                                            c=annotations,
                                            c_mask=src_mask)

        hiddens, ctxs, readout, alignment = results[:4]

        # apply dropout
        if self.dropout < 1.0:
            logger.info('Apply dropout with p = {}'.format(self.dropout))
            readout = Dropout(self.trng, readout, 1, self.dropout)

        p_y_given_x = self.logistic_layer.get_probs(readout)

        self.cost = self.logistic_layer.cost(p_y_given_x, trg, trg_mask) / trg.shape[1]

        # self.cost = theano.printing.Print('likilihood cost:')(self.cost)

        # added by Zhaopeng Tu, 2016-07-12
        # for reconstruction
        if self.with_reconstruction:
            # now hiddens is the annotations
            inverse_init_context = (hiddens * trg_mask[:, :, None]).sum(0) / trg_mask.sum(0)[:, None]

            src_emb = self.table_src.apply(src)
            src_emb_shifted = T.zeros_like(src_emb)
            src_emb_shifted = T.set_subtensor(src_emb_shifted[1:], src_emb[:-1])
            inverse_results = self.inverse_decoder.run_pipeline(state_below=src_emb_shifted,
                                                mask_below=src_mask,
                                                init_context=inverse_init_context,
                                                c=hiddens,
                                                c_mask=trg_mask)

            inverse_hiddens, inverse_ctxs, inverse_readout, inverse_alignment = inverse_results[:4]

            # apply dropout
            if self.dropout < 1.0:
                # logger.info('Apply dropout with p = {}'.format(self.dropout))
                inverse_readout = Dropout(self.srng, inverse_readout, 1, self.dropout)

            p_x_given_y = self.inverse_logistic_layer.get_probs(inverse_readout)

            self.reconstruction_cost = self.inverse_logistic_layer.cost(p_x_given_y, src, src_mask) / src.shape[1]

            # self.reconstruction_cost = theano.printing.Print('reconstructed cost:')(self.reconstruction_cost)
            self.cost += self.reconstruction_cost * self.reconstruction_weight
            

        self.L1 = sum(T.sum(abs(param)) for param in self.params)
        self.L2 = sum(T.sum(param ** 2) for param in self.params)

        params_regular = self.L1 * 1e-6 + self.L2 * 1e-6
        # params_regular = theano.printing.Print('params_regular:')(params_regular)

        # train cost
        train_cost = self.cost + params_regular

        # gradients
        grads = T.grad(train_cost, self.params)

        # apply gradient clipping here
        grads = grad_clip(grads, self.clip_c)

        # train function
        inps = [src, src_mask, trg, trg_mask]
        outs = [train_cost]

        if self.with_layernorm:
            inps = [src, src_mask, trg, trg_mask]
            lr = T.scalar(name='lr')
            print 'Building optimizers...',
            self.train_fn, self.update_fn = adam(lr, self.params, grads, inps, outs)
        else:
            # updates
            updates = adadelta(self.params, grads)

            # mode=theano.Mode(linker='vm') for ifelse 
            # Unless linker='vm' or linker='cvm' are used, ifelse will compute both variables and take the same computation time as switch.
            self.train_fn = theano.function(inps, outs, updates=updates, name='train_function', mode=theano.Mode(linker='vm'))
            # self.train_fn = theano.function(inps, outs, updates=updates, name='train_function', mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))



    def build_sampler(self):

        x = T.lmatrix()

        # Build Networks
        # src_mask is None
        c = self.encoder.apply(x, None)
        #init_context = ctx[0, :, -self.n_hids_src:]
        # mean pooling
        init_context = c.mean(0)

        init_state = self.decoder.create_init_state(init_context)

        # compile function
        print 'Building compile_init_state_and_context function ...'
        self.compile_init_and_context = theano.function([x], [init_state, c],
                                                        name='compile_init_and_context')
        print 'Done'


        y = T.lvector()
        cur_state = T.matrix()

        # if it is the first word, emb should be a1l zero, and it is indicated by -1
        trg_emb = T.switch(y[:, None] < 0,
                           T.alloc(0., 1, self.n_in_trg),
                           self.table_trg.apply(y))

        # added by Zhaopeng Tu, 2016-06-09
        if self.with_coverage:
            cov_before = T.tensor3()
            if self.coverage_type is 'linguistic':
                print 'Building compile_fertility ...'
                fertility = self.decoder._get_fertility(c)
                fertility = T.addbroadcast(fertility,1)
                self.compile_fertility = theano.function([c], [fertility], name='compile_fertility')
                print 'Done'
            else:
                fertility = None
        else:
            cov_before = None
            fertility = None

        # apply one step
        # modified by Zhaopeng Tu, 2016-04-29
        results = self.decoder.apply(state_below=trg_emb,
                                     init_state=cur_state,
                                     c=c,
                                     one_step=True,
                                     # added by Zhaopeng Tu, 2016-04-27
                                     cov_before=cov_before,
                                     fertility=fertility)
        next_state, ctxs, alignment = results[:3]
        idx = 3
        if self.with_coverage:
            cov = results[idx]
            idx += 1

        readout = self.decoder.readout(next_state, ctxs, trg_emb)

        # maxout
        if self.maxout_part > 1:
            readout = self.decoder.one_step_maxout(readout)

        # apply dropout
        if self.dropout < 1.0:
            readout = Dropout(self.trng, readout, 0, self.dropout)

        # compute the softmax probability
        next_probs = self.logistic_layer.get_probs(readout)

        # sample from softmax distribution to get the sample
        next_sample = self.trng.multinomial(pvals=next_probs).argmax(1)

        # compile function
        print 'Building compile_next_state_and_probs function ...'
        inps = [y, cur_state, c]
        outs = [next_probs, next_state, next_sample, alignment]

        # added by Zhaopeng Tu, 2016-04-29
        if self.with_coverage:
            inps.append(cov_before)
            if self.coverage_type is 'linguistic':
                inps.append(fertility)
            outs.append(cov)

        # mode=theano.Mode(linker='vm') for ifelse 
        # Unless linker='vm' or linker='cvm' are used, ifelse will compute both variables and take the same computation time as switch.
        self.compile_next_state_and_probs = theano.function(inps, outs, name='compile_next_state_and_probs', mode=theano.Mode(linker='vm'))
        print 'Done'

        # added by Zhaopeng Tu, 2016-07-18
        # for reconstruction
        if self.with_reconstruction:
            # Build Networks
            # trg_mask is None
            inverse_c = T.tensor3()
            # mean pooling
            inverse_init_context = inverse_c.mean(0)

            inverse_init_state = self.inverse_decoder.create_init_state(inverse_init_context)

            outs = [inverse_init_state]

            # compile function
            print 'Building compile_inverse_init_state_and_context function ...'
            self.compile_inverse_init_and_context = theano.function([inverse_c], outs, name='compile_inverse_init_and_context')
            print 'Done'


            src = T.lvector()
            inverse_cur_state = T.matrix()

            trg_mask = T.matrix()
            # if it is the first word, emb should be all zero, and it is indicated by -1
            src_emb = T.switch(src[:, None] < 0,
                               T.alloc(0., 1, self.n_in_src),
                               self.table_src.apply(src))

            # apply one step
            # modified by Zhaopeng Tu, 2016-04-29
            inverse_results = self.inverse_decoder.apply(state_below=src_emb,
                                         init_state=inverse_cur_state,
                                         c=inverse_c,
                                         c_mask=trg_mask,
                                         one_step=True)
            inverse_next_state, inverse_ctxs, inverse_alignment = inverse_results[:3]

            inverse_readout = self.inverse_decoder.readout(inverse_next_state, inverse_ctxs, src_emb)

            # maxout
            if self.maxout_part > 1:
                inverse_readout = self.inverse_decoder.one_step_maxout(inverse_readout)

            # apply dropout
            if self.dropout < 1.0:
                inverse_readout = Dropout(self.srng, inverse_readout, 0, self.dropout)

            # compute the softmax probability
            inverse_next_probs, inverse_next_energy = self.inverse_logistic_layer.get_probs(inverse_readout)

            # sample from softmax distribution to get the sample
            inverse_next_sample = self.srng.multinomial(pvals=inverse_next_probs).argmax(1)

            # compile function
            print 'Building compile_inverse_next_state_and_probs function ...'
            inps = [src, trg_mask, inverse_cur_state, inverse_c]
            outs = [inverse_next_probs, inverse_next_state, inverse_next_sample, inverse_alignment]

            self.compile_inverse_next_state_and_probs = theano.function(inps, outs, name='compile_inverse_next_state_and_probs')
            print 'Done'



    def save(self, path=None):
        if path is None:
            path = self.path
        filenpz = open(path, "w")
        val = dict([(value.name, value.get_value()) for index, value in enumerate(self.params)])
        logger.info("save the model {}".format(path))
        numpy.savez(path, **val)
        filenpz.close()


    def load(self, path=None):
        if path is None:
            path = self.path
        if os.path.isfile(path):
            logger.info("load params {}".format(path))
            val = numpy.load(path)
            for index, param in enumerate(self.params):
                logger.info('Loading {} with shape {}'.format(param.name, param.get_value(borrow=True).shape))
                if param.name not in val.keys():
                    logger.info('Adding new param {} with shape {}'.format(param.name, param.get_value(borrow=True).shape))
                    continue
                if param.get_value().shape != val[param.name].shape:
                    logger.info("Error: model param != load param shape {} != {}".format(\
                                        param.get_value().shape, val[param.name].shape))
                    raise Exception("loading params shape mismatch")
                else:
                    param.set_value(val[param.name], borrow=True)
        else:
            logger.error("file {} does not exist".format(path))
            self.save()
