mkdir -p $1

GHOG=`pwd`

SAMPLE="python $GHOG/sample.py --beam-search --beam-size 10 --verbose"
THEANO_FLAGS="on_unused_input=ignore,device=gpu2,floatX=float32" $SAMPLE --source=$GHOG/eval/nist_src.txt --trans $1/nist_tran.txt --state $1/search_state.pkl $1/search_model.npz 1>$1/nist.log.txt 2>$1/nist.err.txt
#    878 dev_src.txt
#   1082 tst_src.txt
#   1664 dev_src.txt
#   1357 tst_src.txt
head -878 $1/nist_tran.txt > $1/nist02_tran.txt
head -1960 $1/nist_tran.txt | tail -1082 > $1/nist05_tran.txt
tail -3021 $1/nist_tran.txt | head -1664 > $1/nist06_tran.txt
tail -1357 $1/nist_tran.txt > $1/nist08_tran.txt
wait


./eval/plain2sgm.py $1/nist02_tran.txt eval/nist02_src.sgm $1/nist02_tran.sgm
./eval/mteval-v11b.pl -r eval/nist02_ref.sgm -s eval/nist02_src.sgm -t $1/nist02_tran.sgm > $1/nist02_tran.bleu
 
./eval/plain2sgm.py $1/nist05_tran.txt eval/nist05_src.sgm $1/nist05_tran.sgm
./eval/mteval-v11b.pl -r eval/nist05_ref.sgm -s eval/nist05_src.sgm -t $1/nist05_tran.sgm > $1/nist05_tran.bleu
 
./eval/plain2sgm.py $1/nist06_tran.txt eval/nist06_src.sgm $1/nist06_tran.sgm
./eval/mteval-v11b.pl -r eval/nist06_ref.sgm -s eval/nist06_src.sgm -t $1/nist06_tran.sgm > $1/nist06_tran.bleu
 
./eval/plain2sgm.py $1/nist08_tran.txt eval/nist08_src.sgm $1/nist08_tran.sgm
./eval/mteval-v11b.pl -r eval/nist08_ref.sgm -s eval/nist08_src.sgm -t $1/nist08_tran.sgm > $1/nist08_tran.bleu
 
