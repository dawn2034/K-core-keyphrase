
# %%
# coding=utf-8
import sys
import os
from io import open
from collections import defaultdict, Counter
import networkx as nx
import regex as re
import pickle
import json
import pkuseg
from tqdm import tqdm
import emoji
import math


# %%
def ngrams(lst, n=4):
    N = len(lst)
    if N < n:
        return
        yield

    for i in range(N-n+1):
        yield lst[i: i+n]


def removeSelfLoops(G):
    '''删除自环边'''
    nodes_with_selfloops = G.nodes_with_selfloops()
    for node in nodes_with_selfloops:
        G.remove_edge(node, node)
    return G


def removeSingletons(G):
    '''删除没有边单独的节点'''
    degrees = dict(G.degree())
    for node in degrees.keys():
        if degrees[node] == 0:
            G.remove_node(node)
    return G


def getCoreNumbers(Gpassed):
    '''计算 k-core number'''
    G = Gpassed.copy()
    G = removeSelfLoops(G)
    G = removeSingletons(G)

    degrees = dict(G.degree())
    dmax = max(degrees.values()) + 2
    corenumbers = {}

    N = len(degrees.values())
    for corenumber_k in range(2, dmax):
        whilec = 0
        while min(degrees.values()) < corenumber_k:
            whilec += 1
            for node in degrees.keys():  # remove all the nodes with degrees < k
                if degrees[node] < corenumber_k:
                    corenumbers[node] = corenumber_k - 1
                    G.remove_node(node)

            degrees = dict(G.degree())
            if len(degrees.values()) == 0:
                break

        if len(degrees.values()) == 0:
            break

    return corenumbers


def getCoreRank(G, core_number):
    '''The CoreRank number of a node is defined as 
    the sum of the core numbers of its neighbors. '''
    core_ranks = {}
    for node in list(G.keys()):
        core_ranks[str(node)] = 0
        for nbr in list(G[node].keys()):
            core_ranks[str(node)] += core_number[nbr]
    core_ranks = dict(sorted(core_ranks.items(), key=lambda x: x[1], reverse=True))
    return core_ranks


def inverse_document_frequencies(wordlist, documents):
    idf_values = {}
    for tkn in tqdm(wordlist):
        contains_token = map(lambda doc: tkn in doc, documents)  # 01 list shows the token is in each sentence or not
#         if int(sum(contains_token)) == 0:
#             print(tkn)  # emoji name
        idf_values[tkn] = 1 + math.log(len(documents)/float(sum(contains_token) + 1))
    return idf_values


def tw_idf(core_rank_score, idf):
    twidf = {}
    for w in idf.keys():
        twidf[w] = core_rank_score[w] * idf[w]
    twidf = dict(sorted(twidf.items(), key=lambda x: x[1], reverse=True))
    return twidf


# %%
with open(".//data//neike_all.json", "r", encoding='utf-8') as f:
    dialogs = json.load(f)

useless = ["谢谢", "感谢", "感激", "不客气", "早日康复", "祝你", "满意"]
pucs = '''’'#＜&〰＇『％～—（「\.｠–~？\[|〚｟〟〛』）>_〙„／\$｜＋\/\^、；｣"\*\\,〘“〔〿@〃＂】：｀‘＠:\]\)\}＞【＾〞〖!\{;！」‛\(－=＄‟［”＃〜﹏＆｢〾…＼〕｡`｝＝\+］《＿＊〗\?〝，<､》%。｛‧'''
re_punctuation = "[{}]+".format(pucs)

fw = open(".//data//neike_sentences.txt", "w", encoding='utf-8')

dlg_set = []
for dlg in tqdm(dialogs.values()):
    if dlg not in dlg_set:
        dlg_set.append(dlg)
        for sen in dlg:
            sen = sen.replace("患者:", "")
            sen = sen.replace("医生:", "")
            sen = re.sub(re_punctuation, " ", sen)  # 标点符号替换成空格
            if(len(sen) > 5):  # 过滤过短句子
                cnt = 0
                for w in useless:
                    if w in sen:
                        cnt = 1   # 过滤无用句子
                if(cnt == 0):
                    fw.write(sen + "\n")
fw.close()


# %%
window_size = 4

input_edges_fname = f".//data//neike_all_input_edges_win{window_size}.txt"
fw = open(input_edges_fname, 'w', encoding='utf-8')

with open('.//data//medical_dict.json', 'r', encoding='utf-8') as f:
    user_dict = json.load(f)
seg = pkuseg.pkuseg(user_dict=user_dict)

with open('.//data//stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.readlines()
stopwords = set([w.replace("\n", "") for w in stopwords])

with open(".//data//neike_sentences.txt", 'r', encoding='utf-8') as f:
    sens = f.readlines()
    sens = [sen.replace("\n", "") for sen in sens]

for sen in tqdm(sens):
    sen = emoji.demojize(string=sen)
    tokens = seg.cut(sen)
    tokens = [w for w in tokens if w not in stopwords]
    if(len(tokens) < 5):
        continue
    for ngram in ngrams(tokens, n=window_size):
        for nbr in ngram[1:]:
            fw.write(ngram[0] + ' ' + nbr + '\n')

fw.close()


graphpath = input_edges_fname
G = nx.read_edgelist(graphpath, delimiter=' ', nodetype=str)
gadj = G.adj

core_number = getCoreNumbers(G)

core_rank_score = getCoreRank(G.adj, core_number)

with open(".//data//neike_core_rank_socres.json", 'w', encoding='utf-8') as f:
    json.dump(fp=f, obj=core_rank_score, indent=2, ensure_ascii=False)

wordlist = list(core_number.keys())
idf = inverse_document_frequencies(wordlist, sens)

twidf = tw_idf(core_rank_score, idf)

with open(".//data//neike_twidf.json", 'w', encoding='utf-8') as f:
    json.dump(fp=f, obj=twidf, indent=2, ensure_ascii=False)
