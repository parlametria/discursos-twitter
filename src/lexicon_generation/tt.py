import pandas as pd
import numpy as np
import sys
import os
import random

from statsmodels import robust
from scipy import stats

from sklearn.model_selection import KFold
from sklearn import ensemble
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.linear_model import Lasso

from sklearn.feature_extraction.text import CountVectorizer

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

import unicodedata
import re
import os
import io
import time

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def regex_links(s):
    s = re.sub("(Http|Https|http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"," ",s)
    return s

def preprocess_sentence(w):
    w = regex_links(w)
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z0-9#$]+", " ", w)

    w = w.rstrip().strip()

    #w = ' ' + w + ' '
    return w

rep = {
    " a hora vai chegar ":" a_hora_vai_chegar ",
    " algumas pessoas ": " algumas_pessoas ",
    " acredita se ": " acredita_se ",
    " diz se ": " diz_se ",
    " bem comum ":" bem_comum ",
    " orgao responsavel ":" orgao_responsavel ",
    " muito menos ":" muito_menos ",
    " mundo inteiro ":" mundo_inteiro ",
    " questao de seguranca ":" questao_de_seguranca ",
    " bom senso ":" bom_senso ",
    " velha midia ":" velha_midia ",
    " velha politica ":" velha_politica ",
    " podridao vermelha ":" podridao_vermelha ",
    " coisas erradas ":" coisas_erradas ",
    " novo pais ":" novo_pais ",
    " nova nacao ":" nova_nacao ",
    " isso dai ":" isso_dai ",
    " politica que esta ai ":" politica_que_esta_ai ",
    " em torno de ":" em_torno_de ",
    " que mais ":" que_mais ",
    " que menos ":" que_menos ",
    " a ponto ":" a_ponto ",
    " ao menos ":" ao_menos ",
    " ate mesmo ":" ate_mesmo ",
    " nao mais que ":" nao_mais_que ",
    " nem mesmo ":" nem_mesmo ",
    " no minimo ":" no_minimo ",
    " o unico ":" o_unico ",
    " a unica ":" a_unica ",
    " pelo menos ":" pelo_menos ",
    " quando menos ":" quando_menos ",
    " quando muito ":" quando_muito ",
    " a par disso ":" a_par_disso ",
    " e nao ":" e_nao ",
    " em suma ":" em_suma ",
    " mas tambem ":" mas_tambem ",
    " nao so ":" nao_so ",
    " por sinal ":" por_sinal ",
    " com isso ":" com_isso ",
    " como consequencia ":" como_consequencia ",
    " de modo que ":" de_modo_que ",
    " deste modo ":" deste_modo ",
    " em decorrencia ":" em_decorrencia ",
    " nesse sentido ":" nesse_sentido ",
    " por causa ":" por_causa ",
    " por conseguinte ":" por_conseguinte ",
    " por essa razao ":" por_essa_razao ",
    " por isso ":" por_isso ",
    " sendo assim ":" sendo_assim ",
    " ou entao ":" ou_entao ",
    " ou mesmo ":" ou_mesmo ",
    " como se ":" como_se ",
    " de um lado ":" de_um_lado ",
    " por outro lado ":" por_outro_lado ",
    " mais que ":" mais_que ",
    " menos que ":" menos_que ",
    " desde que ":" desde_que ",
    " do contrario ":" do_contrario ",
    " em lugar ":" em_lugar ",
    " em vez ":" em_vez ",
    " no caso ":" no_caso ",
    " se acaso ":" se_acaso ",
    " de certa forma ":" de_certa_forma ",
    " desse modo ":" desse_modo ",
    " em funcao ":" em_funcao ",
    " isso e ":" isso_e ",
    " ja que ":" ja_que ",
    " na medida que ":" na_medida_que ",
    " nessa direcao ":" nessa_direcao ",
    " no intuito ":" no_intuito ",
    " no mesmo sentido ":" no_mesmo_sentido ",
    " ou seja ":" ou_seja ",
    " uma vez que ":" uma_vez_que ",
    " tanto que ":" tanto_que ",
    " visto que ":" visto_que ",
    " ainda que ":" ainda_que ",
    " ao contrario ":" ao_contrario ",
    " apesar de ":" apesar_de ",
    " fora isso ":" fora_isso ",
    " mesmo que ":" mesmo_que ",
    " nao obstante ":" nao_obstante ",
    " nao fosse isso ":" nao_fosse_isso ",
    " no entanto ":" no_entanto ",
    " para tanto ":" para_tanto ",
    " pelo contrario ":" pelo_contrario ",
    " por sua vez ":" por_sua_vez ",
    " posto que ":" posto_que ",
    " acabar com ":" acabar_com ",
    " nao se importam ":" nao_se_importam ",
    " nunca se importam ":" nunca_se_importam ",
    " virar o jogo ":" virar_o_jogo",
    " povo brasileiro ":" povo_brasileiro ",
    " imprensa mentirosa ":" imprensa_mentirosa ",
    " nao querem que voce saiba ":" nao_querem_que_voce_saiba ",
    " querem nos ":" querem_nos ",
    " querem fazer ":" querem_fazer ",
    " ma fe ":" ma_fe ",
    " a verdade ":" a_verdade ",
    " temos que ":" temos_que ",
    " massa de manobra ":" massa_de_manobra ",
    " nao da pra acreditar ":" nao_da_pra_acreditar ",
    " sirva de exemplo ": " sirva_de_exemplo ",
    " por exemplo ":" por_exemplo ",
    " e se ":" e_se ",
    " ao mesmo tempo ":" ao_mesmo_tempo ",
    " temos de ":" temos_de ",
    " nos calar ":" nos_calar ",
    " bandido bom e bandido morto ":" bandido_bom_e_bandido_morto ",
    " o bem ": " o_bem ",
    " o mal ": " o_mal ",
    " a mentira ":" a_mentira ",
    " o povo ":" o_povo ",
    " meio ambiente ":" meio_ambiente ",
    " reforma tributaria ":" reforma_tributaria ",
    " reforma administrativa ":" reforma_administrativa "
}

rep = dict((re.escape(k), v) for k, v in rep.items()) 
pattern = re.compile("|".join(rep.keys()))

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def readFuncWords():
    stop_words = set(stopwords.words("portuguese"))
    stop_words.update(['que', 'ate', 'esse', 'de', 'do','essa', 'pro', 'pra', 'oi', 'la'])
    return stop_words

def clean_text(line):
    txt = preprocess_sentence(line)
    txt = pattern.sub(lambda m: rep[re.escape(m.group(0))], txt)
    func_tokens = readFuncWords()
    #text_str = text_str.decode("utf-8")
    text_str = txt.lower().strip()
    tokens = nltk.word_tokenize(text_str,language='portuguese')
    # remove single characters
    tokens = [w for w in tokens if len(w) > 2]
    # remove functional words
    tokens = [w for w in tokens if w not in func_tokens]
    return " ".join(tokens)



# Parameters
LABEL_COLUMN_NAME = 'finalGrade'
UNWANTED_COLUMNS = ['nome_eleitoral','class']

N_FOLDS = 5
RANDOM_STATE = 1

DEFAULT_LGB_PARAMS = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 50,
    "verbose": -1,
    "min_data": 5,
    "boost_from_average": True,
    "random_state": 1
}

LABEL_COLUMN_NAME = 'class'
N_FOLDS = 5
N_TIMES_CV = 1

class naive_classifier:
    def __init__(self):
        self.acc = []
    def fit(self,X,y):
        preds = X > 0
        print(preds)
        return 

def first_iter(dataset, mode ='lgb'):
    print("Phase 1: choosing candidate words:")
    matrix = CountVectorizer(min_df=3)
    X = matrix.fit_transform(dataset).toarray()
    aucs = []
    words = []
    i = 0
    for key, value in matrix.vocabulary_.items():
        words.append(key)
        if mode == 'lgb':
            eval_hist = lgb.cv(DEFAULT_LGB_PARAMS, lgb.Dataset(X[:,value].reshape(len(X),1), label=dataset[LABEL_COLUMN_NAME].values), nfold=N_FOLDS)
            metric = np.mean(eval_hist['auc-mean'])
            aucs.append(metric)
        print("word",i,":",key,',',metric)
        i+=1
    ranking = dict(zip(words, aucs))
    return sorted(ranking, key=ranking.get, reverse=True)

def eval_bootstrap(df, features, md=1):
    #print('entrou eval_bootstrap')
    X = df[features].values
    y = df[LABEL_COLUMN_NAME].values

    aucs = []
    for i in range(1,5):
        eval_hist = lgb.cv(DEFAULT_LGB_PARAMS, lgb.Dataset(X, label=y), nfold=N_FOLDS, seed=i)
        metric = np.mean(eval_hist['auc-mean'])
        aucs.append(metric)
    return np.mean(aucs)

def back_one(df, f, md):
    # print("entrou back one")
    v = 0
    f1 = []
    f2 = []
    for i in f:
        f1.insert(len(f1), i)
        f2.insert(len(f2), i)
    AUC = eval_bootstrap(df, f1, md)
    z = AUC
    for i in f:
        f1.remove(i)
        AUC = eval_bootstrap(df, f1, md)
        print("%s,%f" % (f1,AUC))
        if AUC > z:
            v = 1
            z = AUC
            f2 = []
            for j in f1:
                f2.insert(len(f2), j)
        f1.insert(len(f1), i)
    return v,f2

WORD = re.compile(r'\w+')
def regTokenize(text):
    words = WORD.findall(text)
    return words

def tfidf_filter(dataset, threshold):
    tokens = []
    #print('tokenizing documents...')
    for doc in dataset:
        #doc = clean_text(doc)
        tokenize = regTokenize(doc)
        tokens.append(tokenize)
    #print('creating dictionary...')
    dct = Dictionary(tokens)
    corpus = [dct.doc2bow(line) for line in tokens]
    #print(len(corpus))
    #print('creating tf-idf model...')
    model = TfidfModel(corpus,id2word=dct)
    low_value_words = []
    for bow in corpus:
        low_value_words += [id for id, value in model[bow] if (value < threshold)] #and dct[id] != "reforma_tributaria")]
    #print("low_value_words:",len(low_value_words))
    dct.filter_tokens(bad_ids=low_value_words)
    new_corpus = [dct.doc2bow(doc) for doc in tokens]
    #print(len(new_corpus))
    corp = []
    for doc in new_corpus:
        corp.append([dct[id] for id, value in doc])
    return corp


def expand_dataset(original_df, df_atual ,new_feature,vocab):
    print("new_feature=",new_feature)
    df['menciona_RT'] = original_df.text.str.contains(new_feature)
    df_false = df[df.menciona_RT == 0]
    df_true = df[df.menciona_RT == 1]
    df_false = df_false.sample(len(df_true))
    new_df = pd.concat([df_atual,df_true,df_false])
    df_list = new_df.text.to_list()
    corpus = tfidf_filter(df_list, 0.055)
    new_corpus = []
    for doc in corpus:
        new_corpus.append(" ".join(doc))
    new_df['clean_text'] = new_corpus
    matrix = CountVectorizer(vocabulary=vocab)
    X = matrix.fit_transform(new_df.clean_text).toarray()
    classes = new_df['menciona_RT'].astype(int).to_list()
    new_df_matrix = pd.DataFrame(X, columns=matrix.get_feature_names())

    new_df_matrix['class'] = classes

    return new_df_matrix, new_df




# Reads dataset
df = pd.read_csv(sys.argv[1])
df.dropna(axis=0, subset=['text'], inplace=True)
#df['text'] = df.text.apply(clean_text)
#df[['text']].to_csv('clean_text.csv',index = False)
df['menciona_RT'] = df.text.str.contains("reforma_tributaria|reforma_administrativa")
df_false = df[df.menciona_RT == 0]
df_true = df[df.menciona_RT == 1]
df_false = df_false.sample(len(df_true))
initial_df = pd.concat([df_true,df_false])
df_list = initial_df.text.to_list()
corpus = tfidf_filter(df_list, 0.055)
#for doc in corpus:
    #print(doc)


# if len(sys.argv) < 3:
#     top_words = first_iter(df)[:1000]
#     with open("top_single.txt","w") as f:
#         print("writing best candidates to \"top_single.txt\"...")
#         for word in top_words:
#             f.write(word+"\n")
#     with open("top_single.txt","r") as f:
#         top_words = []
#         for word in f:
#             top_words.append(word[:-1])
# else:
#     with open(sys.argv[2],"r") as f:
#         top_words = []
#         for word in f:
#             top_words.append(word[:-1])

nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()

top_words = []
new_corpus = []
for doc in corpus:
    new_corpus.append(" ".join(doc))
    for word in doc:
        #stemmed = stemmer.stem(word)
        #top_words.append(stemmed)
        top_words.append(word)

initial_df['clean_text'] = new_corpus
top_words = list(set(top_words))
#initial_df['clean_text'].to_csv("test.csv",index = False)

def stem_text(line):
    line = line.split(" ")
    tokens = []
    for string in line:
        #print(string)
        try: tokens.append(stemmer.stem(string))
        except: continue
    return " ".join(tokens)


#initial_df['clean_text'] = initial_df.clean_text.apply(stem_text)
#initial_df['clean_text'].to_csv("test2.csv",index = False)

# nltk.download('rslp')
# stemmer = nltk.stem.RSLPStemmer()
# a = stemmer.stem("presidente")
# print(a)
#print(len(top_words))

matrix = CountVectorizer(vocabulary=top_words)
X = matrix.fit_transform(initial_df.clean_text).toarray()
classes = initial_df['menciona_RT'].astype(int).to_list()
#print(len(classes))
df_matrix = pd.DataFrame(X, columns=matrix.get_feature_names())

#print(df_matrix.shape)
df_matrix['class'] = classes

df_matrix.to_csv("df_matrix.csv",index=False)
#print(X.shape)

RANDOM_STATE = 1
all_features = list(top_words)
if "class" in all_features: all_features.remove("class")
#print(all_features)

md = 1
f = []
i = 0
best_auc = 0
best_auc_lexicon = []

aucs = []
words = []
# print(df_matrix)
#print(all_features)
for feature1 in all_features:
    if i == 20: break
    if feature1 in f: continue
    k = 0
    x = feature1
    i = i + 1
    j = 0
    for feature2 in all_features:
        if feature2 in f: continue
        j = j + 1
        f.insert(len(f), feature2)
        AUC = eval_bootstrap(df_matrix, f, md)
        print("%s,%f" % (f,AUC))
        aucs.append(AUC)
        words.append(feature2)
        z = AUC 
        f.remove(feature2)
        sys.stdout.flush()
        if z > k:
            x = feature2
            k = z
    ranking = dict(zip(words, aucs))
    all_features = sorted(ranking, key=ranking.get, reverse=True)[0:1000]
    f.insert(len(f), x)
    #if x != "reforma_tributaria":
    #    df_matrix,initial_df = expand_dataset(df, initial_df ,x,all_features)
    if i > 2:
        v,f = back_one(df_matrix, f, md)
        while v == 1:
            v,f = back_one(df_matrix, f, md)
        i = len(f)