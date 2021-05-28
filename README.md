# AttentionRank
AttentionRank: Unsupervised keyphrase Extraction using Self and Cross Attentions

Download "stanford-corenlp-full-2018-02-27" and pretrained bert-base from below link:
https://indiana-my.sharepoint.com/:f:/g/personal/hd10_iu_edu/Ep1hNQYehrlMkB734awOKhQBTv3qVVsW8iO8bMl4Vdg46Q?e=0oI0y4

Run "stanford-corenlp-full-2018-02-27" with terminal:
  cd stanford-corenlp-full-2018-02-27/
  java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 &
  
Run python files by their indices.

