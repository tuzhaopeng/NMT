# nmt-adequacy
**Attention-based NMT with Coverage, Context Gate, and Reconstruction**

This is a Theano-based RNNSearch, which integrates the following techniques to alleviate the problem of fluent but inadequate translations that NMT suffers from.

— <a href="http://arxiv.org/abs/1601.04811">**Coverage**</a> to indicate whether a source word is translated or not, which proves to alleviate over-translation and under-translation.

— <a href="http://arxiv.org/abs/1608.06043">**Context Gate**</a> to dynamically control the ratios at which source and target contexts contribute to the generation of target words, which enhances the adequacy of NMT while keeping the fluency unchanged.

— <a href="http://arxiv.org/abs/1611.01874">**Reconstruction**</a> to reconstruct the input source sentence from the hidden layer of the output target sentence, to ensure that the information in the source side is transformed to the target side as much as possible.


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
@InProceedings{Tu:2017:AAAI,
      author    = {Tu, Zhaopeng and Liu, Yang and Shang, Lifeng and Liu, Xiaohua and Li, Hang},
      title     = {Neural Machine Translation with Reconstruction},
      booktitle = {Proceedings of the 31st AAAI Conference on Artificial Intelligence},
      year      = {2017},
}
</code>
</pre>


For any comments or questions, please  email <a href="mailto:tuzhaopeng@gmail.com">the first author</a>.
