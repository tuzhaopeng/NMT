#!/usr/bin/env bash

# Run this script to test that your current code works as it worked before
# Needs environmental variable GHOG to be set to the location of the repository.
#
# At first run will create a directory test_workspace to which test data will
# be downloaded.

GHOG=`pwd`

[ "$DEBUG" = 1 ] && set -x
set -u

STATUS="ok"
#export PYTHONPATH=$GHOG:$PYTHONPATH


#echo "Stage 1: Test scoring"
#SCORE="$GHOG/score.py --mode=batch --src=english.txt --trg=french.txt --allow-unk"
#echo "Score with RNNencdec"
#$SCORE --state encdec_state.pkl encdec_model.npz >encdec_scores.txt 2>>log.txt
#check_result $NUMDIFF encdec_scores.txt "RNNencdec scores changed!"
#echo "Score with RNNsearch"
#$SCORE --state search_state.pkl search_model.npz >search_scores.txt 2>>log.txt
#check_result $NUMDIFF search_scores.txt "RNNsearch scores changed!"

echo "Sample with RNNsearch"
SAMPLE="python $GHOG/sample.py --beam-search --beam-size 10 --verbose"
THEANO_FLAGS="on_unused_input=warn,device=gpu3,floatX=float32" $SAMPLE --source=$GHOG/eval/nist_src.txt --trans nist_tran.txt --state search_state.pkl search_model.npz 1>nist.log.txt 2>nist.err.txt
#    878 dev_src.txt
#   1082 tst_src.txt
#   1664 dev_src.txt
#   1357 tst_src.txt
head -878 nist_tran.txt > nist02_tran.txt
head -1960 nist_tran.txt | tail -1082 > nist05_tran.txt
tail -3021 nist_tran.txt | head -1664 > nist06_tran.txt
tail -1357 nist_tran.txt > nist08_tran.txt
wait
