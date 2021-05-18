# AttentionRank
AttentionRank: Unsupervised keyphrase Extraction using Self and Cross Attentions

Keyword or keyphrase extraction is to identify words or phrases presenting the main topics of a document. This paper proposes the AttentionRank, a hybrid attention model, to identify keyphrase from a document in an unsupervised manner. AttentionRank calculates self-attention and cross-attention using a pre-trained language model. The self-attention is designed to determine the importance of a candidate within the context of a sentence. The cross-attention is to identify the semantic relevance between a candidate and other sentences. 
We evaluate the AttentionRank on three publicly available datasets against six baselines. The results show that the AttentionRank is an effective and robust unsupervised keyphrase extraction model on both long and short documents.
