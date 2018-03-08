# rnn encoder-decoder for machine translation
import numpy
import theano.tensor as T
import argparse
import logging
import pprint
import time
import os
from stream import DStream, get_devtest_stream
from search import BeamSearch
from sampling import Sampler, BleuValidator
from nmt import EncoderDecoder
import configurations


if __name__=='__main__':
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto",  default="get_config_search_coverage",
                        help="Prototype config to use for config")
    # added by Zhaopeng Tu, 2016-05-12
    parser.add_argument("--state", help="State to use")
    # added by Zhaopeng Tu, 2016-07-14
    parser.add_argument("--start", type=int, default=0, help="Iterations to start")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    configuration = getattr(configurations, args.proto)()
    # added by Zhaopeng Tu, 2016-05-12
    if args.state:
        configuration.update(eval(open(args.state).read()))
    logger.info("\nModel options:\n{}".format(pprint.pformat(configuration)))

    batch_size = configuration['batch_size']

    src = T.lmatrix()
    src_mask = T.matrix()
    trg = T.lmatrix()
    trg_mask = T.matrix()

    rng = numpy.random.RandomState(1234)

    enc_dec = EncoderDecoder(rng, **configuration)
    enc_dec.build_trainer(src, src_mask, trg, trg_mask)
    enc_dec.build_sampler()

    if configuration['reload']:
        enc_dec.load()

    sample_search = BeamSearch(enc_dec=enc_dec,
                               configuration=configuration,
                               beam_size=1,
                               maxlen=configuration['seq_len_src'], stochastic=True)
    valid_search = BeamSearch(enc_dec=enc_dec, 
                              configuration=configuration,
                              beam_size=configuration['beam_size'],
                              maxlen=3*configuration['seq_len_src'], stochastic=False)

    sampler = Sampler(sample_search, **configuration)
    bleuvalidator = BleuValidator(valid_search, **configuration)

    # train function
    train_fn = enc_dec.train_fn
    if configuration.get('with_layernorm', False):
        update_fn = enc_dec.update_fn

    # train data
    ds = DStream(**configuration)

    # valid data
    vs = get_devtest_stream(data_type='valid', input_file=None, **configuration)

    # main_loop
    # modified by Zhaopeng Tu, 2016-07-14
    # to continue training
    # iters = 0
    iters = args.start
    valid_bleu_best = -1
    epoch_best = -1
    iters_best = -1
    max_epochs = configuration['finish_after']

    for epoch in range(max_epochs):
        for x, x_mask, y, y_mask in ds.get_iterator():
            last_time = time.time()
            tc = train_fn(x.T, x_mask.T, y.T, y_mask.T)

            # added by Zhaopeng Tu, 2017-11-29
            # for layer_norm with adam, we explicitly update parameters
            # will be merged in the future
            if enc_dec.with_layernorm:
                update_fn(0.001)

            cur_time = time.time()
            iters += 1
            logger.info('epoch %d \t updates %d train cost %.4f use time %.4f'
                        %(epoch, iters, tc[0], cur_time-last_time))

            if iters % configuration['save_freq'] == 0:
                enc_dec.save()

            if iters % configuration['sample_freq'] == 0:
                sampler.apply(x, y)

            if iters < configuration['val_burn_in']:
                continue

            if (iters <= configuration['val_burn_in_fine'] and iters % configuration['valid_freq'] == 0) \
               or (iters > configuration['val_burn_in_fine'] and iters % configuration['valid_freq_fine'] == 0):
                valid_bleu = bleuvalidator.apply(vs, configuration['valid_src'], configuration['valid_out'])
                os.system('mkdir -p out/%d' % iters)
                os.system('mv %s* %s out/%d' % (configuration['valid_out'], configuration['saveto'], iters))
                logger.info('valid_test \t epoch %d \t updates %d valid_bleu %.4f'
                        %(epoch, iters, valid_bleu))
                if valid_bleu > valid_bleu_best:
                    valid_bleu_best = valid_bleu
                    epoch_best = epoch
                    iters_best = iters
                    enc_dec.save(path=configuration['saveto_best'])

    logger.info('final result: epoch %d \t updates %d valid_bleu_best %.4f'
            %(epoch_best, iters_best, valid_bleu_best))

