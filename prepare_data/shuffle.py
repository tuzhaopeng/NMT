import argparse
import numpy

def RunShuffle(source, target, source_shuffle, target_shuffle):

    fin_src = open(source, 'rU')
    fin_trg = open(target, 'rU')
    fout_src = open(source_shuffle, 'w')
    fout_trg = open(target_shuffle, 'w')

    sent_src = []
    sent_trg = []

    for line in fin_src:
        sent_src.append(line.strip())
    for line in fin_trg:
        sent_trg.append(line.strip())

    assert len(sent_src) == len(sent_trg)

    idxs = numpy.arange(len(sent_src))
    numpy.random.shuffle(idxs)

    for i in idxs:
        fout_src.write(sent_src[i]+'\n')
        fout_trg.write(sent_trg[i]+'\n')

    fin_src.close()
    fin_trg.close()
    fout_src.close()
    fout_trg.close()


if __name__ == '__main__':
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str)
    parser.add_argument("target", type=str)
    args = parser.parse_args()

    RunShuffle(args.source, args.target, args.source+'.shuffle', args.target+'.shuffle')
