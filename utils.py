#utils
import numpy
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


# dropout
def Dropout(trng, state_before, use_noise, dropout):
    proj = T.switch(use_noise,
                    state_before * trng.binomial(state_before.shape, p=1.-dropout, n=1,
                                                 dtype=state_before.dtype),
                    state_before * (1.-dropout))
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


def ReplicateLayer(x, n_times):
    a = T.shape_padleft(x)
    padding = [1] * x.ndim
    b = T.alloc(numpy.float32(1), n_times, *padding)
    return a * b


def concatenate(tensor_list, axis=0):
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = T.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# layer normalization
# code from https://github.com/ryankiros/layer-norm

scale_add = 0.0
scale_mul = 1.0

def layer_norm(x, b, s):
    _eps = numpy.float32(1e-5)
    
    if x.ndim == 3:
        output = (x - x.mean(2)[:,:,None]) / T.sqrt((x.var(2)[:,:,None] + _eps))
        output = s[None, None, :] * output + b[None, None,:]
    else:
        output = (x - x.mean(1)[:,None]) / T.sqrt((x.var(1)[:,None] + _eps))
        output = s[None, :] * output + b[None,:]
    return output


