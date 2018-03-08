# models: BidirectionalEncoder, Decoder, GRU, LogisticRegression, LookupTable
import numpy
import theano
import theano.tensor as T
from initialization import constant_weight, uniform_weight, ortho_weight, norm_weight
from theano.tensor.nnet import categorical_crossentropy
from utils import ReplicateLayer, _p, concatenate, layer_norm, scale_add, scale_mul


class BidirectionalEncoder(object):

    def __init__(self, rng, n_in, n_hids, table, name='rnn_encoder'):

        # lookup table
        self.table = table
        # embedding dimension
        self.n_in = n_in
        # hidden state dimension
        self.n_hids = n_hids

        self.params = []
        self.layers = []

        self.forward = GRU(rng, self.n_in, self.n_hids, name=_p(name, 'forward'))
        self.layers.append(self.forward)

        self.backward = GRU(rng, self.n_in, self.n_hids, name=_p(name, 'backward'))
        self.layers.append(self.backward)

        for layer in self.layers:
            self.params.extend(layer.params)


    def apply(self, sentence, sentence_mask):

        state_below = self.table.apply(sentence)

        # make sure state_below: n_steps * batch_size * embedding
        if state_below.ndim == 2:
            n_steps = state_below.shape[0]
            embed = state_below.shape[1]
            state_below = state_below.reshape((n_steps, 1, embed))

        hiddens_forward = self.forward.apply(state_below, sentence_mask)

        if sentence_mask is None:
            hiddens_backward = self.backward.apply(state_below[::-1])
        else:
            hiddens_backward = self.backward.apply(state_below[::-1], sentence_mask[::-1])

        training_c_components = []
        training_c_components.append(hiddens_forward)
        training_c_components.append(hiddens_backward[::-1])

        #annotaitons = T.concatenate(training_c_components, axis=2)
        annotaitons = concatenate(training_c_components, axis=training_c_components[0].ndim-1)

        return annotaitons


class Decoder(object):

    def __init__(self, rng, n_in, n_hids, n_cdim, maxout_part=2,
                 name='rnn_decoder',
                 # added by Zhaopeng Tu, 2016-04-29
                 with_coverage=False, coverage_dim=1, coverage_type='linguistic', max_fertility=2, 
                 # added by Zhaopeng Tu, 2016-05-30
                 with_context_gate=False,
                 # added by Zhaopeng Tu, 2017-11-28
                 with_layernorm=False):

        self.n_in = n_in
        self.n_hids = n_hids
        self.n_cdim = n_cdim
        self.maxout_part = maxout_part
        self.pname = name
        # added by Zhaopeng Tu, 2016-04-29
        self.with_coverage = with_coverage
        self.coverage_dim = coverage_dim
        assert coverage_type in ['linguistic', 'neural'], 'Coverage type must be either linguistic or neural'
        self.coverage_type = coverage_type
        self.max_fertility = max_fertility
        # added by Zhaopeng Tu, 2016-05-30
        self.with_context_gate=with_context_gate
        # added by Zhaopeng Tu, 2017-11-28
        self.with_layernorm = with_layernorm
        ##################################
        self.rng = rng

        self._init_params()

    def _init_params(self):

        shape_xh = (self.n_in, self.n_hids)
        shape_hh = (self.n_hids, self.n_hids)

        self.W_xz = norm_weight(rng=self.rng, shape=shape_xh, name=_p(self.pname, 'W_xz'))
        self.W_xr = norm_weight(rng=self.rng, shape=shape_xh, name=_p(self.pname, 'W_xr'))
        self.W_xh = norm_weight(rng=self.rng, shape=shape_xh, name=_p(self.pname, 'W_xh'))
        self.b_z = constant_weight(shape=(self.n_hids, ), name=_p(self.pname, 'b_z'))
        self.b_r = constant_weight(shape=(self.n_hids, ), name=_p(self.pname, 'b_r'))
        self.b_h = constant_weight(shape=(self.n_hids, ), name=_p(self.pname, 'b_h'))
        self.W_hz = ortho_weight(rng=self.rng, shape=shape_hh, name=_p(self.pname, 'W_hz'))
        self.W_hr = ortho_weight(rng=self.rng, shape=shape_hh, name=_p(self.pname, 'W_hr'))
        self.W_hh = ortho_weight(rng=self.rng, shape=shape_hh, name=_p(self.pname, 'W_hh'))

        self.params = [self.W_xz, self.W_xr, self.W_xh,
                       self.W_hz, self.W_hr, self.W_hh,
                       self.b_z, self.b_r, self.b_h]

        if self.with_layernorm:
            self.W_hz_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'hz_lnb'))
            self.W_hz_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'hz_lns'))
            self.W_hr_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'hr_lnb'))
            self.W_hr_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'hr_lns'))
            self.W_hh_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'hh_lnb'))
            self.W_hh_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'hh_lns'))
            self.params += [self.W_hz_lnb, self.W_hz_lns, 
                            self.W_hr_lnb, self.W_hr_lns, 
                            self.W_hh_lnb, self.W_hh_lns]


        shape_ch = (self.n_cdim, self.n_hids)
        self.W_cz = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_cz'))
        self.W_cr = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_cr'))
        self.W_ch = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_ch'))
        self.W_c_init = norm_weight(rng=self.rng, shape=(self.n_cdim, self.n_hids), name=_p(self.pname, 'W_c_init'))
        self.b_c_init = constant_weight(shape=(self.n_hids, ), name=_p(self.pname, 'b_c_init'))

        self.params += [self.W_cz, self.W_cr, self.W_ch, 
                        self.W_c_init, self.b_c_init]

        # commented by Zhaopeng Tu, 2016-04-29
        # modification in this version
        # in the paper, e_{i,j} = a(s_{i-1}, h_j)
        # here, e_{i,j} = a(GRU(s_{i-1}, y_{i-1}), h_j), which considers the lastly generated target word
        # all the following parameters are for the introduced GRU
        # it is reasonable
        self.W_n1_h = ortho_weight(rng=self.rng, shape=shape_hh, name=_p(self.pname, 'W_n1_h'))
        self.W_n1_r = ortho_weight(rng=self.rng, shape=shape_hh, name=_p(self.pname, 'W_n1_r'))
        self.W_n1_z = ortho_weight(rng=self.rng, shape=shape_hh, name=_p(self.pname, 'W_n1_z'))
        self.b_n1_h = constant_weight(shape=(self.n_hids, ), name=_p(self.pname, 'b_n1_h'))
        self.b_n1_r = constant_weight(shape=(self.n_hids, ), name=_p(self.pname, 'b_n1_r'))
        self.b_n1_z = constant_weight(shape=(self.n_hids, ), name=_p(self.pname, 'b_n1_z'))
        self.params += [self.W_n1_h, self.W_n1_r, self.W_n1_z, 
                        self.b_n1_h, self.b_n1_r, self.b_n1_z]
        ###############################################
        if self.with_layernorm:
            self.W_n1_z_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'W_n1_z_lnb'))
            self.W_n1_z_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'W_n1_z_lns'))
            self.W_n1_r_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'W_n1_r_lnb'))
            self.W_n1_r_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'W_n1_r_lns'))
            self.W_n1_h_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'W_n1_h_lnb'))
            self.W_n1_h_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'W_n1_h_lns'))
        
            self.c_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'c_lnb'))
            self.c_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'c_lns'))
            #self.W_z_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'z_lnb'))
            #self.W_z_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'z_lns'))

            self.params += [self.W_n1_z_lnb, self.W_n1_z_lns, 
                            self.W_n1_r_lnb, self.W_n1_r_lns, 
                            self.W_n1_h_lnb, self.W_n1_h_lns, 
                            self.c_lnb, self.c_lns]


        
        # for attention model
        self.A_cp = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'A_cp'))
        self.B_hp = norm_weight(rng=self.rng, shape=shape_hh, name=_p(self.pname, 'B_hp'))
        self.b_tt = constant_weight(shape=(self.n_hids, ), name=_p(self.pname, 'b_tt'))
        self.D_pe = norm_weight(rng=self.rng, shape=(self.n_hids, 1), name=_p(self.pname, 'D_pe'))
        self.c_tt = constant_weight(shape=(1, ), name=_p(self.pname, 'c_tt'))

        self.params += [self.A_cp, self.B_hp, self.b_tt, 
                        self.D_pe, self.c_tt]


        # added by Zhaopeng Tu, 2016-04-29
        # coverage only works for attention model
        if self.with_coverage:
            shape_covh = (self.coverage_dim, self.n_hids)
            self.C_covp = norm_weight(rng=self.rng, shape=shape_covh, name=_p(self.pname, 'Cov_covp'))
            self.params.append(self.C_covp)

            if self.coverage_type is 'linguistic':
                # for linguistic coverage, fertility model is necessary since it yields better translation and alignment quality
                self.W_cov_fertility = norm_weight(rng=self.rng, shape=(self.n_cdim, 1), name=_p(self.pname, 'W_cov_fertility'))
                self.b_cov_fertility = constant_weight(shape=(1, ), name=_p(self.pname, 'b_cov_fertility'))
                self.params += [self.W_cov_fertility, self.b_cov_fertility]
            else:
                # for neural network based coverage, gating is necessary
                shape_covcov = (self.coverage_dim, self.coverage_dim)
                self.W_cov_h = ortho_weight(rng=self.rng, shape=shape_covcov, name=_p(self.pname, 'W_cov_h'))
                self.W_cov_r = ortho_weight(rng=self.rng, shape=shape_covcov, name=_p(self.pname, 'W_cov_r'))
                self.W_cov_z = ortho_weight(rng=self.rng, shape=shape_covcov, name=_p(self.pname, 'W_cov_z'))
                self.b_cov_h = constant_weight(shape=(self.coverage_dim, ), name=_p(self.pname, 'b_cov_h'))
                self.b_cov_r = constant_weight(shape=(self.coverage_dim, ), name=_p(self.pname, 'b_cov_r'))
                self.b_cov_z = constant_weight(shape=(self.coverage_dim, ), name=_p(self.pname, 'b_cov_z'))

                self.params += [self.W_cov_h, self.W_cov_r, self.W_cov_z, 
                                self.b_cov_h, self.b_cov_r, self.b_cov_z]

                # added by Zhaopeng Tu, 2017-11-29
                if self.with_layernorm:
                    self.W_cov_z_lnb = constant_weight(shape=(self.coverage_dim), value=scale_add, name=_p(self.pname, 'cov_z_lnb'))
                    self.W_cov_z_lns = constant_weight(shape=(self.coverage_dim), value=scale_mul, name=_p(self.pname, 'cov_z_lns'))
                    self.W_cov_r_lnb = constant_weight(shape=(self.coverage_dim), value=scale_add, name=_p(self.pname, 'cov_r_lnb'))
                    self.W_cov_r_lns = constant_weight(shape=(self.coverage_dim), value=scale_mul, name=_p(self.pname, 'cov_r_lns'))
                    self.W_cov_h_lnb = constant_weight(shape=(self.coverage_dim), value=scale_add, name=_p(self.pname, 'cov_h_lnb'))
                    self.W_cov_h_lns = constant_weight(shape=(self.coverage_dim), value=scale_mul, name=_p(self.pname, 'cov_h_lns'))
                    self.params += [self.W_cov_z_lnb, self.W_cov_z_lns, 
                                    self.W_cov_r_lnb, self.W_cov_r_lns, 
                                    self.W_cov_h_lnb, self.W_cov_h_lns]


                # parameters for coverage inputs
                # attention probablity
                self.W_cov_ph = norm_weight(rng=self.rng, shape=(1, self.coverage_dim), name=_p(self.pname, 'W_cov_ph'))
                self.W_cov_pr = norm_weight(rng=self.rng, shape=(1, self.coverage_dim), name=_p(self.pname, 'W_cov_pr'))
                self.W_cov_pz = norm_weight(rng=self.rng, shape=(1, self.coverage_dim), name=_p(self.pname, 'W_cov_pz'))
                # source annotations
                self.W_cov_ch = norm_weight(rng=self.rng, shape=(self.n_cdim, self.coverage_dim), name=_p(self.pname, 'W_cov_ch'))
                self.W_cov_cr = norm_weight(rng=self.rng, shape=(self.n_cdim, self.coverage_dim), name=_p(self.pname, 'W_cov_cr'))
                self.W_cov_cz = norm_weight(rng=self.rng, shape=(self.n_cdim, self.coverage_dim), name=_p(self.pname, 'W_cov_cz'))
                # previous decoding states
                self.W_cov_hh = norm_weight(rng=self.rng, shape=(self.n_hids, self.coverage_dim), name=_p(self.pname, 'W_cov_hh'))
                self.W_cov_hr = norm_weight(rng=self.rng, shape=(self.n_hids, self.coverage_dim), name=_p(self.pname, 'W_cov_hr'))
                self.W_cov_hz = norm_weight(rng=self.rng, shape=(self.n_hids, self.coverage_dim), name=_p(self.pname, 'W_cov_hz'))
                
                self.params += [self.W_cov_ph, self.W_cov_pr, self.W_cov_pz, self.W_cov_ch, self.W_cov_cr, self.W_cov_cz, self.W_cov_hh, self.W_cov_hr, self.W_cov_hz]
        ####################################################

        # added by Zhaopeng Tu, 2016-05-30
        # for context gate, which works for both with_attention and with_context modes
        if self.with_context_gate:
            # parameters for coverage inputs
            # input form target context
            self.W_ctx_h = norm_weight(rng=self.rng, shape=(self.n_hids, self.n_hids), name=_p(self.pname, 'W_ctx_h'))
            self.W_ctx_c = norm_weight(rng=self.rng, shape=(self.n_cdim, self.n_hids), name=_p(self.pname, 'W_ctx_c'))
            self.b_ctx = constant_weight(shape=(self.n_hids, ), name=_p(self.pname, 'b_ctx'))
            self.params += [self.W_ctx_h, self.W_ctx_c]

        # for readout
        n_out = self.n_in * self.maxout_part
        self.W_o_c = norm_weight(rng=self.rng, shape=(self.n_cdim, n_out), name=_p(self.pname, 'W_out_c'))
        self.W_o_h = norm_weight(rng=self.rng, shape=(self.n_hids, n_out), name=_p(self.pname, 'W_out_h'))
        self.W_o_e = norm_weight(rng=self.rng, shape=(self.n_in, n_out), name=_p(self.pname, 'W_out_e'))
        self.b_o = constant_weight(shape=(n_out, ), name=_p(self.pname, 'b_out_o'))
        self.params += [self.W_o_c, self.W_o_h, self.W_o_e, self.b_o]


    #################### coverage model #####################
    # added by Zhaopeng Tu, 2016-04-29
    # for fertility model
    def _get_fertility(self, c):
        fertility = T.nnet.sigmoid(T.dot(c, self.W_cov_fertility) + self.b_cov_fertility) * self.max_fertility
        fertility = fertility.reshape((c.shape[0], c.shape[1]))
        return fertility


    def _update_coverage(self, cov_tm1, probs, c, h_tm1, fertility=None):
        '''
        cov_tm1:    coverage at time (t-1)
        probs:      attention probabilities at time t
        c:          source annotations
        fertility:  fertility of individual source word
        '''
        if self.coverage_type is 'linguistic':
            assert fertility, 'ferility should be given for linguistic coverage'
            fertility_probs = probs/fertility
            cov = T.unbroadcast(fertility_probs.dimshuffle(0,1,'x'), 2)
            
            # accumulation
            cov = cov_tm1 + cov
        else:
            # we can precompute w*c in advance to minimize the computational cost
            extend_probs = probs.dimshuffle(0,1,'x')
            
            if self.with_layernorm:
                z = layer_norm((T.dot(cov_tm1, self.W_cov_z) + T.dot(extend_probs, self.W_cov_pz) + T.dot(c, self.W_cov_cz) + T.dot(h_tm1, self.W_cov_hz) + self.b_cov_z), self.W_cov_z_lnb, self.W_cov_z_lns)
                z = T.nnet.sigmoid(z)
                r = layer_norm((T.dot(cov_tm1, self.W_cov_r) + T.dot(extend_probs, self.W_cov_pr) + T.dot(c, self.W_cov_cr) + T.dot(h_tm1, self.W_cov_hr) + self.b_cov_r), self.W_cov_r_lnb, self.W_cov_r_lns)
                r = T.nnet.sigmoid(r)
                cov = layer_norm((r * T.dot(cov_tm1, self.W_cov_h) + T.dot(extend_probs, self.W_cov_ph) + T.dot(c, self.W_cov_ch) + T.dot(h_tm1, self.W_cov_hh) + self.b_cov_h), self.W_cov_h_lnb, self.W_cov_h_lns)
                cov = T.tanh(cov)
            else:
                z = T.nnet.sigmoid(T.dot(cov_tm1, self.W_cov_z) + T.dot(extend_probs, self.W_cov_pz) + T.dot(c, self.W_cov_cz) + T.dot(h_tm1, self.W_cov_hz) + self.b_cov_z)
                r = T.nnet.sigmoid(T.dot(cov_tm1, self.W_cov_r) + T.dot(extend_probs, self.W_cov_pr) + T.dot(c, self.W_cov_cr) + T.dot(h_tm1, self.W_cov_hr) + self.b_cov_r)
                cov = T.tanh(r * T.dot(cov_tm1, self.W_cov_h) + T.dot(extend_probs, self.W_cov_ph) + T.dot(c, self.W_cov_ch) + T.dot(h_tm1, self.W_cov_hh) + self.b_cov_h)

            cov = (1-z) * cov_tm1 + z * cov

        return cov
    #################### coverage model #####################


    def _step_attention(self, x_h, x_z, x_r, x_m, h_tm1, c, c_m, p_from_c, cov_tm1=None, fertility=None):
        '''
        x_h: input at time t
        x_z: update of input
        x_r: reset of input
        x_m: mask of x_t
        h_tm1: previous state
        # added by Zhaopeng Tu, 2016-04-29
        cov_tm1:  coverage at time (t-1)
        fertility:  fertility of individual source word
        '''

        # for attention model
        source_len = c.shape[0]
        target_num = h_tm1.shape[0]

        # commented by Zhaopeng Tu, 2016-04-29
        # here h1 combines previous hidden state and lastly generated word with GRU
        # note that this is different from the paper
        if self.with_layernorm:
            z1 = layer_norm((T.dot(h_tm1, self.W_n1_z) + x_z + self.b_n1_z), self.W_n1_z_lnb, self.W_n1_z_lns)
            z1 = T.nnet.sigmoid(z1)
            r1 = layer_norm((T.dot(h_tm1, self.W_n1_r) + x_r + self.b_n1_r), self.W_n1_r_lnb, self.W_n1_r_lns)
            r1 = T.nnet.sigmoid(r1)
            h1 = layer_norm((r1 * T.dot(h_tm1, self.W_n1_h) + x_h + self.b_n1_h), self.W_n1_h_lnb, self.W_n1_h_lns)
            h1 = T.tanh(h1)
        else:
            z1 = T.nnet.sigmoid(T.dot(h_tm1, self.W_n1_z) + x_z + self.b_n1_z)
            r1 = T.nnet.sigmoid(T.dot(h_tm1, self.W_n1_r) + x_r + self.b_n1_r)
            h1 = T.tanh(r1 * T.dot(h_tm1, self.W_n1_h) + x_h + self.b_n1_h)

        h1 = z1 * h_tm1 + (1. - z1) * h1
        h1 = x_m[:, None] * h1 + (1. - x_m)[:, None] * h_tm1

        p_from_h = ReplicateLayer(T.dot(h1, self.B_hp), source_len)
        p = p_from_h + p_from_c + self.b_tt

        # added by Zhaopeng Tu, 2016-04-29
        if self.with_coverage:
            p_from_cov = T.dot(cov_tm1, self.C_covp)
            p += p_from_cov

        energy = T.exp(T.dot(T.tanh(p), self.D_pe) + self.c_tt).reshape((source_len, target_num))
        if c_m:
            energy *= c_m

        normalizer = energy.sum(axis=0, keepdims=True)
        probs = energy / normalizer

        ctx = (c * probs.dimshuffle(0, 1, 'x')).sum(axis=0)

        # added by Zhaopeng Tu, 2016-04-29
        # update coverage after producing attention probabilities at time t
        if self.with_coverage:
            cov = self._update_coverage(cov_tm1, probs, c, h_tm1, fertility)

        # commented by Zhaopeng Tu, 2016-04-29
        # this is even more consistent with our context gate
        # h1 corresponds to target context, while ctx corresponds to source context
        # added by Zhaopeng Tu, 2016-05-30
        if self.with_context_gate:
            gate = T.nnet.sigmoid(T.dot(h1, self.W_ctx_h) +
                                  T.dot(ctx, self.W_ctx_c) + self.b_ctx)
            
            # we directly scale h1, since it used in computing both can_h_t and h_t
            h1 = h1 * (1.-gate)
        else:
            gate = 1.

        # modified by Zhaopeng Tu, 2017-11-28
        if self.with_layernorm:
            z_t = layer_norm((T.dot(h1, self.W_hz) + T.dot(ctx, self.W_cz) + self.b_z), self.W_hz_lnb, self.W_hz_lns)
            z_t = T.nnet.sigmoid(z_t)
            r_t = layer_norm((T.dot(h1, self.W_hr) + T.dot(ctx, self.W_cr) + self.b_r), self.W_hr_lnb, self.W_hr_lns)
            r_t = T.nnet.sigmoid(r_t)
            h_t = layer_norm((r_t * T.dot(h1, self.W_hh) + T.dot(ctx, self.W_ch) + self.b_h), self.W_hh_lnb, self.W_hh_lns)
            h_t = T.tanh(h_t)
        else:
            z_t = T.nnet.sigmoid(T.dot(h1, self.W_hz) + gate * T.dot(ctx, self.W_cz) + self.b_z)
            r_t = T.nnet.sigmoid(T.dot(h1, self.W_hr) + gate * T.dot(ctx, self.W_cr) + self.b_r)
            h_t = T.tanh(r_t * T.dot(h1, self.W_hh) + gate * T.dot(ctx, self.W_ch) + self.b_h)

        h_t = z_t * h1 + (1. - z_t) * h_t
        h_t = x_m[:, None] * h_t + (1. - x_m[:, None]) * h1
     
        results = [h_t, ctx, probs]

        if self.with_coverage:
            results += [cov]

        return results


    def create_init_state(self, init_context):
        init_state = T.tanh(T.dot(init_context, self.W_c_init)+self.b_c_init)
        return init_state


    def apply(self, state_below, mask_below=None, init_state=None,
              init_context=None, c=None, c_mask=None, one_step=False,
              # added by Zhaopeng Tu, 2016-04-29
              cov_before=None, fertility=None):

        # assert c, 'Context must be provided'
        # assert c.ndim == 3, 'Context must be 3-d: n_seq * batch_size * dim'

        # state_below: n_steps * batch_size/1 * embedding
        if state_below.ndim == 3:
            n_steps = state_below.shape[0]
            batch_size = state_below.shape[1]
        else:
            batch_size = 1

        # mask
        if mask_below is None: #sampling or beamsearch
            mask_below = T.alloc(numpy.float32(1.), state_below.shape[0], 1)

        if one_step:
            assert init_state, 'previous state mush be provided'

        if init_state is None:
            init_state = self.create_init_state(init_context)
        
        state_below_xh = T.dot(state_below, self.W_xh)
        state_below_xz = T.dot(state_below, self.W_xz)
        state_below_xr = T.dot(state_below, self.W_xr)

        # for attention model
        p_from_c = T.dot(c, self.A_cp).reshape((c.shape[0], c.shape[1], self.n_hids))
        if self.with_layernorm:
            p_from_c = layer_norm(p_from_c, self.c_lnb, self.c_lns)

        if one_step:
            return self._step_attention(state_below_xh, state_below_xz, state_below_xr, \
                                        mask_below, init_state, c, c_mask, p_from_c, \
                                        # added by Zhaopeng Tu, 2016-06-08
                                        cov_tm1=cov_before, fertility=fertility)
        else:
            sequences = [state_below_xh, state_below_xz, state_below_xr, mask_below]
            # decoder hidden state
            outputs_info = [init_state]
            non_sequences = [c, c_mask, p_from_c]
            # added by Zhaopeng Tu, 2016-04-29
            # ctx, probs
            outputs_info += [None, None]
            if self.with_coverage:
                # initialization for coverage
                init_cov = T.unbroadcast(T.zeros((c.shape[0], c.shape[1], self.coverage_dim), dtype='float32'), 2)
                outputs_info.append(init_cov)
                
                # fertility is not constructed outside when training
                if self.coverage_type is 'linguistic':
                    fertility = self._get_fertility(c)
                else:
                    fertility = T.zeros((c.shape[0], c.shape[1]), dtype='float32')
                non_sequences.append(fertility)

            # modified by Zhaopeng Tu, 2016-05-02
            # rval, updates = theano.scan(self._step_attention,
            if not self.with_coverage:
                             # seqs              |  out    |   non_seqs
                fn = lambda  x_h, x_z, x_r, x_m,    h_tm1,     c, c_m, p_from_c :  self._step_attention(x_h, x_z, x_r, x_m, h_tm1, c, c_m, p_from_c)
            else:
                             # seqs              |  out              |   non_seqs
                fn = lambda  x_h, x_z, x_r, x_m,    h_tm1, cov_tm1,      c, c_m, p_from_c, fertility :  self._step_attention(x_h, x_z, x_r, x_m, h_tm1, c, c_m, p_from_c, cov_tm1=cov_tm1, fertility=fertility)

            rval, updates = theano.scan(fn,
                                    sequences=sequences,
                                    non_sequences=non_sequences,
                                    # outputs_info=[init_state, None],
                                    outputs_info=outputs_info,
                                    name=_p(self.pname, 'layers'),
                                    n_steps=n_steps)

            self.output = rval

            return self.output


    def readout(self, hiddens, ctxs, state_below):

        readout = T.dot(hiddens, self.W_o_h) + \
                  T.dot(ctxs, self.W_o_c) + \
                  T.dot(state_below, self.W_o_e) + \
                  self.b_o

        return T.tanh(readout)


    def one_step_maxout(self, readout):

        maxout = readout.reshape((readout.shape[0],
                                  readout.shape[1]/self.maxout_part,
                                  self.maxout_part), ndim=3).max(axis=2)

        return maxout


    def run_pipeline(self, state_below, mask_below, init_context=None, c=None, c_mask=None)

        init_state = self.create_init_state(init_context)

        # modified by Zhaopeng Tu, 2016-04-29
        # [hiddens, ctxs] = self.apply(state_below=state_below, mask_below=mask_below,
        results = self.apply(state_below=state_below, mask_below=mask_below,
                             init_state=init_state, c=c, c_mask=c_mask)
        hiddens, ctxs, probs = results[:3]
        idx = 3
        if self.with_coverage:
            covs = results[idx]
            idx += 1
        
        # readout
        readout = self.readout(hiddens, ctxs, state_below)

        # maxout
        if self.maxout_part > 1:
            readout = readout.reshape((readout.shape[0], \
                                       readout.shape[1], \
                                       readout.shape[2]/self.maxout_part, \
                                       self.maxout_part), \
                                      ndim=4).max(axis=3)

        # modified by Zhaopeng Tu, 2016-07-12
        # for reconstruction, we need decoder states
        # return readout * mask_below[:, :, None]
        results = [hiddens, ctxs, readout * mask_below[:, :, None], probs]
        return results



class LookupTable(object):

    def __init__(self, rng, vocab_size, embedding_size, name='embeddings'):

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # for norm_weight
        self.W = norm_weight(rng=rng, shape=(vocab_size, embedding_size), name=name)

		# parameters of the model
        self.params = [self.W]


    def apply(self, indices):

        outshape = [indices.shape[i] for i in range(indices.ndim)] + [self.embedding_size]
        return self.W[indices.flatten()].reshape(outshape)


class LogisticRegression(object):

    """Multi-class Logistic Regression Class"""

    def __init__(self, rng, n_in, n_out, name='LR'):

        # initialize the weights W as a matrix of shape (n_in, n_out)
        self.W = norm_weight(rng=rng, shape=(n_in, n_out), name=_p(name, 'W'))

		# initialize the baises b as a vector of n_out 0s
        self.b = constant_weight(shape=(n_out, ), name=_p(name, 'b'))

        # parameters of the model
        self.params = [self.W, self.b]


    def get_probs(self, input):
        # compute vector of class-membership probabilities in symbolic form
        energy = T.dot(input, self.W) + self.b

        if energy.ndim == 3:
            energy_exp = T.exp(energy - T.max(energy, 2, keepdims=True))
            p_y_given_x = energy_exp / energy_exp.sum(2, keepdims=True)
        else:
            p_y_given_x = T.nnet.softmax(energy)

        return p_y_given_x


    def cost(self, p_y_given_x, targets, mask=None):
        prediction = p_y_given_x
        if prediction.ndim == 3:
            prediction_flat = prediction.reshape(((prediction.shape[0] *
                                                   prediction.shape[1]),
                                                  prediction.shape[2]), ndim=2)
            targets_flat = targets.flatten()
            mask_flat = mask.flatten()
            ce = categorical_crossentropy(prediction_flat, targets_flat) * mask_flat

            return T.sum(ce)

        assert mask is None
        ce = categorical_crossentropy(prediction, targets)
        return T.sum(ce)


    def errors(self, y, p_y_given_x):
        y_pred = T.argmax(p_y_given_x, axis=-1)
        if y.ndim == 2:
            y = y.flatten()
            y_pred = y_pred.flatten()

        return T.sum(T.neq(y, y_pred))


class GRU(object):

    def __init__(self, rng, n_in, n_hids, name='GRU', with_context=False, with_layernorm=False):

        self.n_in = n_in
        self.n_hids = n_hids
        self.pname = name
        self.rng = rng

        self.with_context = with_context
        if self.with_context:
            self.c_hids = n_hids
        
        self.with_layernorm = with_layernorm

        self._init_params()


    def _init_params(self):

        shape_xh = (self.n_in, self.n_hids)
        shape_hh = (self.n_hids, self.n_hids)

        self.W_xz = norm_weight(rng=self.rng, shape=shape_xh, name=_p(self.pname, 'W_xz'))
        self.W_xr = norm_weight(rng=self.rng, shape=shape_xh, name=_p(self.pname, 'W_xr'))
        self.W_xh = norm_weight(rng=self.rng, shape=shape_xh, name=_p(self.pname, 'W_xh'))
        self.b_z = constant_weight(shape=(self.n_hids, ), name=_p(self.pname, 'b_z'))
        self.b_r = constant_weight(shape=(self.n_hids, ), name=_p(self.pname, 'b_r'))
        self.b_h = constant_weight(shape=(self.n_hids, ), name=_p(self.pname, 'b_h'))
        self.W_hz = ortho_weight(rng=self.rng, shape=shape_hh, name=_p(self.pname, 'W_hz'))
        self.W_hr = ortho_weight(rng=self.rng, shape=shape_hh, name=_p(self.pname, 'W_hr'))
        self.W_hh = ortho_weight(rng=self.rng, shape=shape_hh, name=_p(self.pname, 'W_hh'))

        self.params = [self.W_xz, self.W_xr, self.W_xh,
                       self.W_hz, self.W_hr, self.W_hh,
                       self.b_z, self.b_r, self.b_h]

        if self.with_layernorm:
            self.W_xz_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'xz_lnb'))
            self.W_xz_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'xz_lns'))
            self.W_xr_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'xr_lnb'))
            self.W_xr_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'xr_lns'))
            self.W_xh_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'xh_lnb'))
            self.W_xh_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'xh_lns'))

            self.W_z_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'z_lnb'))
            self.W_z_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'z_lns'))
            self.W_r_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'r_lnb'))
            self.W_r_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'r_lns'))
            self.W_h_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'h_lnb'))
            self.W_h_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'h_lns'))

            self.params += [self.W_xz_lnb, self.W_xz_lns, self.W_xr_lnb, self.W_xr_lns, self.W_xh_lnb, self.W_xh_lns, \
                           self.W_z_lnb, self.W_z_lns, self.W_r_lnb, self.W_r_lns, self.W_h_lnb, self.W_h_lns] 


        if self.with_context:
            shape_ch = (self.c_hids, self.n_hids)
            self.W_cz = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_cz'))
            self.W_cr = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_cr'))
            self.W_ch = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_ch'))
            self.W_c_init = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_c_init'))

            self.params += [self.W_cz, self.W_cr, self.W_ch, self.W_c_init]

            if self.with_layernorm:
                self.W_cz_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'cz_lnb'))
                self.W_cz_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'cz_lns'))
                self.W_cr_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'cr_lnb'))
                self.W_cr_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'cr_lns'))
                self.W_ch_lnb = constant_weight(shape=(self.n_hids), value=scale_add, name=_p(self.pname, 'ch_lnb'))
                self.W_ch_lns = constant_weight(shape=(self.n_hids), value=scale_mul, name=_p(self.pname, 'ch_lns'))

                self.params += [self.W_cz_lnb, self.W_cz_lns, self.W_cr_lnb, self.W_cr_lns, self.W_ch_lnb, self.W_ch_lns]


    def _step(self, x_h, x_z, x_r, x_m, h_tm1):
        '''
        x_h: input at time t
        x_z: update for x_t
        x_r: reset for x_t
        x_m: mask of x_t
        h_tm1: previous state
        '''

        if self.with_layernorm:
            z_t = layer_norm((x_z + T.dot(h_tm1, self.W_hz) + self.b_z), self.W_z_lnb, self.W_z_lns)
            z_t = T.nnet.sigmoid(z_t)                                                                     

            r_t = layer_norm((x_r + T.dot(h_tm1, self.W_hr) + self.b_r), self.W_r_lnb, self.W_r_lns)
            r_t = T.nnet.sigmoid(r_t)      

            can_h_t = layer_norm((x_h + r_t * T.dot(h_tm1, self.W_hh) + self.b_h), self.W_h_lnb, self.W_h_lns)
            can_h_t = T.tanh(can_h_t)
        else:
            z_t = T.nnet.sigmoid(x_z + T.dot(h_tm1, self.W_hz) + self.b_z)
            r_t = T.nnet.sigmoid(x_r + T.dot(h_tm1, self.W_hr) + self.b_r)
            can_h_t = T.tanh(x_h + r_t * T.dot(h_tm1, self.W_hh) + self.b_h)

        h_t = (1. - z_t) * h_tm1 + z_t * can_h_t

        h_t = x_m[:, None] * h_t + (1. - x_m[:, None]) * h_tm1

        return h_t


    def _step_context(self, x_t, x_m, h_tm1, cz, cr, ch):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        '''

        if self.with_layernorm:
            z_t = layer_norm((T.dot(x_t, self.W_xz) +
                                 T.dot(h_tm1, self.W_hz) +
                                 T.dot(cz, self.W_cz) + self.b_z), self.W_z_lnb, self.W_z_lns)
            z_t = T.nnet.sigmoid(z_t)

            r_t = layer_norm((T.dot(x_t, self.W_xr) +
                                 T.dot(h_tm1, self.W_hr) +
                                 T.dot(cr, self.W_cr) + self.b_r), self.W_r_lnb, self.W_r_lns)
            r_t = T.nnet.sigmoid(r_t)

            can_h_t = layer_norm((T.dot(x_t, self.W_xh) +
                             r_t * T.dot(h_tm1, self.W_hh) +
                             T.dot(ch, self.W_ch) + self.b_h), self.W_h_lnb, self.W_h_lns)
            can_h_t = T.tanh(can_h_t)
        else:
            z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) +
                                 T.dot(h_tm1, self.W_hz) +
                                 T.dot(cz, self.W_cz) + self.b_z)

            r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) +
                                 T.dot(h_tm1, self.W_hr) +
                                 T.dot(cr, self.W_cr) + self.b_r)

            can_h_t = T.tanh(T.dot(x_t, self.W_xh) +
                             r_t * T.dot(h_tm1, self.W_hh) +
                             T.dot(ch, self.W_ch) + self.b_h)

        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t

        h_t = x_m[:, None] * h_t + (1. - x_m[:, None])*h_tm1

        return h_t


    def apply(self, state_below, mask_below=None, init_state=None, context=None):

        n_steps = state_below.shape[0]
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
        else:
            batch_size = 1
            state_below = state_below.reshape((n_steps, batch_size, state_below.shape[1]))

        if mask_below is None:
            mask_below = T.alloc(numpy.float32(1.), n_steps, 1)

        if self.with_context:
            assert context

            if init_state is None:
                init_state = T.tanh(T.dot(context, self.W_c_init))

            c_z = T.dot(context, self.W_cz)
            c_r = T.dot(context, self.W_cr)
            c_h = T.dot(context, self.W_ch)
            if self.with_layernorm:
                c_h = layer_norm(c_h, self.W_ch_lnb, self.W_ch_lns)
                c_z = layer_norm(c_z, self.W_cz_lnb, self.W_cz_lns)
                c_r = layer_norm(c_r, self.W_cr_lnb, self.W_cr_lns)

            non_sequences = [c_z, c_r, c_h]

            rval, updates = theano.scan(self._step_context,
                                        sequences=[state_below, mask_below],
                                        non_sequences=non_sequences,
                                        outputs_info=[init_state],
                                        name=_p(self.pname, 'layers'),
                                        n_steps=n_steps)
        else:
            if init_state is None:
                init_state = T.alloc(numpy.float32(0.), batch_size, self.n_hids)

            state_below_xh = T.dot(state_below, self.W_xh)
            state_below_xz = T.dot(state_below, self.W_xz)
            state_below_xr = T.dot(state_below, self.W_xr)

            if self.with_layernorm:
                state_below_xh = layer_norm(state_below_xh, self.W_xh_lnb, self.W_xh_lns)
                state_below_xz = layer_norm(state_below_xz, self.W_xz_lnb, self.W_xz_lns)
                state_below_xr = layer_norm(state_below_xr, self.W_xr_lnb, self.W_xr_lns)

            sequences = [state_below_xh, state_below_xz, state_below_xr, mask_below]

            rval, updates = theano.scan(self._step,
                                        sequences=sequences,
                                        outputs_info=[init_state],
                                        name=_p(self.pname, 'layers'),
                                        n_steps=n_steps)
        self.output = rval

        return self.output


    def run_pipeline(self, state_below, mask_below, context=None):

        hiddens = self.apply(state_below, mask_below, context=context)

        if self.with_context:
            n_in = self.n_in + self.n_hids + self.c_hids
            n_out = self.n_hids * 2
            n_times = state_below.shape[0]
            r_context = ReplicateLayer(context, n_times)
            combine = T.concatenate([state_below, hiddens, r_context], axis=2)
        else:
            n_in = self.n_in + self.n_hids
            n_out = self.n_hids * 2 # for maxout
            combine = T.concatenate([state_below, hiddens], axis=2)

        self.W_m = norm_weight(rng=self.rng, shape=(n_in, n_out), name=_p(self.pname, 'W_m'))
        self.b_m = constant_weight(rng=self.rng, shape=(n_out,), name=_p(self.pname, 'b_m'))

        self.params += [self.W_m, self.b_m]

        # maxout
        merge_out = theano.dot(combine, self.W_m) + self.b_m
        merge_max_out = merge_out.reshape((merge_out.shape[0],
                                       merge_out.shape[1],
                                       merge_out.shape[2]/2,
                                       2), ndim=4).max(axis=3)

        return merge_max_out * mask_below[:, :, None]

