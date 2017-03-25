# NMT
**Attention-based NMT with Coverage and Context Gate**


We are still in the process of releasing our neural machine translation (NMT) code, which alleviates the problem of fluent but inadequate translations that NMT suffers.
In this version, we introduce:

— <a href="http://arxiv.org/abs/1601.04811">**Coverage**</a> to indicate whether a source word is translated or not, which proves to alleviate over-translation and under-translation.

— <a href="http://arxiv.org/abs/1608.06043">**Context Gate**</a> to dynamically control the ratios at which source and target contexts contribute to the generation of target words, which enhances the adequacy of NMT while keeping the fluency unchanged.

Using coverage mechanism significantly improves upon a standard attention-based NMT system by +1.8 BLEU, and incorporating context gate obtains a further improvement of +1.6 BLEU (i.e., **+3.4 BLEU** in total).

If you use the code, please cite our papers:
<pre>
<code>
@InProceedings{Tu:2016:ACL,
      author    = {Tu, Zhaopeng and Lu, Zhengdong and Liu, Yang and Liu, Xiaohua and Li, Hang},
      title     = {Modeling Coverage for Neural Machine Translation},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics},
      year      = {2016},
}
@Article{Tu:2017:TACL,
      author    = {Tu, Zhaopeng and Liu, Yang and Lu, Zhengdong and Liu, Xiaohua and Li, Hang},
      title     = {Context Gates for Neural Machine Translation},
      booktitle = {Transactions of the Association for Computational Linguistics},
      year      = {2017},
}
</code>
</pre>


For any comments or questions, please  email <a href="mailto:tuzhaopeng@gmail.com">the first author</a>.


Installation
------------

NMT is developed by <a href="http://www.zptu.net">Zhaopeng Tu</a>, which is on top of <a href="https://github.com/lisa-groundhog/GroundHog">lisa-groudhog</a>. It requires Theano0.8 or above version (for the module "scan" used in the trainer).

To install NMT in a multi-user setting

``python setup.py develop --user``

For general installation, simply use

``python setup.py develop``

NOTE: This will install the development version of Theano, if Theano is not currently installed.


How to Run?
--------------------------

See experiments/nmt/README.md
