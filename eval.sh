THEANO_FLAGS="device=$2,floatX=float32" python sampling.py --state config.py --model $1/model.npz ./data/test_src ./data/test_trg $1/test_out 1>$1/test.log 2>$1/test.err

