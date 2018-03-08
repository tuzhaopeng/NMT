#!/usr/bin/python 

import sys, os
import numpy

def read_dict(dict_file):
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

    return word_dict


def get_hard_align(align):
    align_matrix = numpy.array(eval(align))
    # hard accuracy, extract only one-to-one alignments by selecting the source_words word with the highest weight per target word
    max_probs = align_matrix.max(axis=0)
    align_matrix = align_matrix >= max_probs

    return align_matrix


if len(sys.argv) != 4:
    print './replace_unk.py input dict output'
    sys.exit(-1)

fin = open(sys.argv[1])
fout = open(sys.argv[3], 'w')

word_dict = read_dict(sys.argv[2])

while 1:
    try:
        line = fin.next().strip()
    except StopIteration:
        break

    if line.startswith('Parsed Input'):
        source_words = line[13:].strip().split()
        tran_words = fin.next()[12:].strip().split()
        fin.next()
        align = fin.next().strip()
        align = get_hard_align(align)
        
        new_tran_words = []
        for i in xrange(len(tran_words)):
            if tran_words[i] != 'UNK':
                new_tran_words.append(tran_words[i])
            else:
                # scan the column and get the source word with highest probability
                aligned_source_index = -1
                probs = align[:,i]
                for j in xrange(len(probs)):
                    if probs[j] == 1:
                        aligned_source_index = j
                        break

                if aligned_source_index < 0:
                    print 'Error'
                    print probs
                    sys.exit(1)

                aligned_source_word = source_words[aligned_source_index]

                if aligned_source_word == '<eos>':
                    continue
                elif aligned_source_word in word_dict:
                    # replace the UNK with translation from dictionary
                    new_tran_words.append(word_dict[aligned_source_word][0])
                else:
                    # we put the source word to the translation
                    new_tran_words.append(aligned_source_word)

        print >> fout, ' '.join(new_tran_words)


