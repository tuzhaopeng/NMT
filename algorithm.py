# learning algorithms
import numpy
import theano
import theano.tensor as T
from itertools import izip


def adadelta(parameters, gradients, rho=0.95, eps=1e-6, delta_bias=0.):
    # create variables to store intermediate updates
    gradients_sq = [theano.shared(numpy.zeros(p.get_value().shape,
    							  dtype=theano.config.floatX))
    			    for p in parameters]
    deltas_sq = [theano.shared(numpy.zeros(p.get_value().shape,
    						   dtype=theano.config.floatX))
    		    for p in parameters]

    # calculates the new "average" delta for the next iteration
    gradients_sq_new = [rho*g_sq + (1-rho)*(g**2)
    				   for g_sq,g in izip(gradients_sq, gradients)]

    # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
    # deltas = [(T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad
    deltas = [(T.sqrt(d_sq+eps)/(T.sqrt(g_sq+eps)+delta_bias))*grad
    		 for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients)]

    # added by Zhaopeng Tu, 2017-05-19
    # remove nan when the step is too big
    # deltas = [T.switch(T.isnan(delta)+T.isinf(delta), eps, delta) for delta in deltas]
    # deltas = [T.clip(delta, -100, 100) for delta in deltas]

    # calculates the new "average" deltas for the next step.
    deltas_sq_new = [rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas)]

    # Prepare it as a list f
    gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
    deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
    # parameters_updates = [(p,T.clip(p - d, -15,15)) for p,d in izip(parameters,deltas)]
    parameters_updates = [(p,p-d) for p,d in izip(parameters,deltas)]

    return gradient_sq_updates + deltas_sq_updates + parameters_updates


# added by Zhaopeng Tu, 2017-11-29
# for layer_norm
profile = False

# modification from dl4mt by shaohui kuang 2017-09-23
def adam(lr, tparams, grads, inp, out, beta1=0.9, beta2=0.999, e=1e-8):
    gshared = [theano.shared(p.get_value() * 0.) for p in tparams]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, out, updates=gsup, profile=profile, name='train_function', mode=theano.Mode(linker='vm'))

    updates = []

    t_prev = theano.shared(numpy.float32(0.))
    t = t_prev + 1.
    lr_t = lr * T.sqrt(1. - beta2**t) / (1. - beta1**t)

    for p, g in zip(tparams, gshared):
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')
        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g**2
        step = lr_t * m_t / (T.sqrt(v_t) + e)
        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))

    #return updates
    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile, name='train_function', mode=theano.Mode(linker='vm'))

    return f_grad_shared, f_update


def grad_clip(grads, clip_c):
    # apply gradient clipping
    if clip_c > 0:
        g2 = 0.

        for g in grads:
            g2 += (g**2).sum()

        new_grads = []
        for g in grads:
            new_grads.append(T.switch(g2 > (clip_c ** 2),
                             g / T.sqrt(g2) * clip_c,
                             g))
        grads = new_grads

    return grads

