# AttentionRank

Update 20221002: Fixed a computing mistake in step 011. Line 103 and line 131, move out the calculations of min and max from the loop.

This is an implement of paper AttentionRank: Unsupervised keyphrase Extraction using Self and Cross Attentions

Keyphrase Extractor can be run as below:

1, Download and extract all files.

2, Download "stanford-corenlp-full-2018-02-27" and pretrained bert-base from below link:

    https://indiana-my.sharepoint.com/:f:/g/personal/hd10_iu_edu/Ep1hNQYehrlMkB734awOKhQBTv3qVVsW8iO8bMl4Vdg46Q?e=0oI0y4

3, Run "stanford-corenlp-full-2018-02-27" with terminal:

    (1) cd stanford-corenlp-full-2018-02-27/
    
    (2) java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 &
  
4, Run python files by their indices (001 - 011).

