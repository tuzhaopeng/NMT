mkdir -p $1
./eval/test.bash
mv nist* $1

./eval/plain2sgm.py $1/nist02_tran.txt eval/nist02_src.sgm $1/nist02_tran.sgm
./eval/mteval-v11b.pl -r eval/nist02_ref.sgm -s eval/nist02_src.sgm -t $1/nist02_tran.sgm > $1/nist02_tran.bleu
 
./eval/plain2sgm.py $1/nist05_tran.txt eval/nist05_src.sgm $1/nist05_tran.sgm
./eval/mteval-v11b.pl -r eval/nist05_ref.sgm -s eval/nist05_src.sgm -t $1/nist05_tran.sgm > $1/nist05_tran.bleu
 
./eval/plain2sgm.py $1/nist06_tran.txt eval/nist06_src.sgm $1/nist06_tran.sgm
./eval/mteval-v11b.pl -r eval/nist06_ref.sgm -s eval/nist06_src.sgm -t $1/nist06_tran.sgm > $1/nist06_tran.bleu
 
./eval/plain2sgm.py $1/nist08_tran.txt eval/nist08_src.sgm $1/nist08_tran.sgm
./eval/mteval-v11b.pl -r eval/nist08_ref.sgm -s eval/nist08_src.sgm -t $1/nist08_tran.sgm > $1/nist08_tran.bleu
 
