# K-core-keyphrase
K-core keyphrase

按照《Unsupervised Abstractive Meeting Summarization with Multi-Sentence Compression and Budgeted Submodular Maximization》ACL2018
的Graph-based word importance scoring计算词权重。

1. 处理语料
    对春雨医生的心内科对话语料 xinneike.json 作去除标点符号(标点符号替换成空格)、去emoji表情、短句子过滤等处理，得到sentences.txt

2. 构建词图
    对sentences.txt读取句子，用pkuseg分词，过滤停用词。设置窗口大小，对每个句子做滑动窗口取ngram，[w1,w2,w3,w4]，成边w1-w2, w1-w3, w1-w4。将两个词w1 w2作为边写入到input_edges_win4.txt的每一行。
    import networkx 包，载入input_edges构成无向图得到Word co-occurrence network。

3. 计算k-core number、 CoreRank score
    将图中所有度小于k的节点删除后留下的子图，其core number 为k，子图的每个节点赋予k
    CoreRank score 即每个节点将其所有相邻节点的core number加和。最后计算每个词对整段对话的IDF，最后对每个词 CoreRank*IDF 得到词权重
