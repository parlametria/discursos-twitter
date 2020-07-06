"""
CODE ADAPTED FROM https://github.com/HongyuGong/Document-Similarity-via-Hidden-Topics

Hongyu Gong, Tarek Sakakini, Suma Bhat and Jinjun Xiong, “Document Similarity for Texts of Varying Lengths via Hidden Topics”,
accepted by the 56th Annual Meeting of the Association for Computational Linguistics (ACL), 2018.

main.py
long_docs: list of concateation of reviewers' works

short_docs: list of abstract in submissions

matching_scores: shape (submission_num, reviewers)
each entry (i,j) is a real value in range of (0,1),
indicating how well submission i matches reviewer j
"""
import pandas as pd
import numpy as np
import preprocess
import tfidf_helper
import match_helper
import re
from gensim.models import KeyedVectors
import argparse


def vectorizeShortDoc(raw_docs, word_vectors, is_refine=False, word_limit=100):
    """
    word vectors for each short doc
    """
    # tokenize
    print("vectorize short docs...")
    docs = []
    for raw_doc in raw_docs:
        try:
            docs.append(preprocess.tokenizeText(raw_doc))
        except:
            continue
    #print(docs[0:5])
    #docs = preprocess.tokenizeText(raw_docs)
    if (is_refine):
        docs = tfidf_helper.extract(docs, word_limit)
    docs_vecs = match_helper.findWordVectors(docs, word_vectors)
    return docs_vecs
    


def vectorizeLongDoc(raw_docs, word_vectors, topic_num=10, is_refine=False, word_limit=100):
    """
    raw_docs: a list of the concateation of reviewers' works
    vector space for each long doc
    """
    # tokenize
    print("vectorize long docs...")
    docs = []
    for raw_doc in raw_docs:
        docs.append(preprocess.tokenizeText(raw_doc))
    #docs = preprocess.tokenizeText(raw_docs)
    # if refine with tf-idf methods
    if (is_refine):
        docs = tfidf_helper.extract(docs, word_limit)
    docs_topics, topic_weights = match_helper.findHiddenTopics(docs, word_vectors, topic_num)
    return docs_topics, topic_weights
    


def mapping(embedding_path, raw_short_docs, raw_long_docs, topic_num=10, \
            is_binary=False, is_refine_short=False, is_refine_long=False, \
            short_word_limit=100, long_word_limit=1000):
    # load word embeddings
    print('loading word vectors...')
    word_vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=is_binary, limit=None)
    # tokenize long docs
    #print('vectorize_short')
    short_docs_vecs = vectorizeShortDoc(raw_short_docs, word_vectors, \
                                        is_refine_short, short_word_limit)
    #print('Vectorize lexicons')
    long_docs_vecs, topic_weights = vectorizeLongDoc(raw_long_docs, word_vectors, topic_num, \
                                      is_refine_long, long_word_limit)

    #print('weighted matching')
    matching_scores = match_helper.weightedMatching(short_docs_vecs, long_docs_vecs, topic_weights)
    #print('end')
    matching_scores = np.sqrt(matching_scores)
    return matching_scores

def extract_texts(input_txt):
    df = pd.read_csv(input_txt)
    df['clean_tweets'] = [re.sub(r"(?:\@|https?\://)\S+", '', str(x)) for x in df['text']]
    df['clean_tweets'] = [re.sub("(\\d|\\W)+|\w*\d\w*"," ", str(x)) for x in df['clean_tweets']]
    df['clean_tweets'] = [re.sub(r"\b[a-zA-Z]\b", "", str(x)) for x in df['clean_tweets']]
    texts = df['clean_tweets'].tolist()
    return texts

def read_lexicons():
    outrage = "mamata desinformacao absurdo denunciar compartilhar sanguinario revolta vandalismo desrespeito desordem caos ahoravaichegar enganar engane guerra acabarcom naoseimportam nuncaseimportam perde inimigo querem ataques virarojogo contra agora bandidagem ladrao vagabundos povobrasileiro corruptos horror circo safados ordem imprensamentirosa naoqueremquevocesaiba canalhas queremnos queremfazer mafe desmascarado averdade temosque ratos esgoto reagir calar lixo escoria mentirosa corja roubalheira porca inadimissivel inaceitavel massademanobra naodapraacreditar"
    vagueza = "coisa alguns gente frequentemente provavelmente algumaspessoas dizse frequentemente provavelmente bemcomum sustentavel integrado acho claramente dito noticiado talvez especialistas considerado acreditase bandido pouco muito orgaoresponsavel diversos sempre nunca muitomenos jamais autoridades responsaveis varias mundointeiro outros acham bastante propria questaodeseguranca bomsenso teria urgente classe mestre informacoes situacao  dificil facil demonstrou corruptos local velhamidia velhapolitica podridaovermelha iria viria faria homem mulher coisaserradas novopais novanacao issodai politicaqueestaai questao vicios  politicos elemento nos eles nada verdade inverdade mentira quem emtornode ninguem indio quemais quemenos mal bem vitimizacao nenhum algum"
    argumentacao = "incluindo inclusive mesmo aponto aomenos apenas ate atemesmo incluindo inclusive mesmo naomaisque nemmesmo nominimo ounico aunica pelomenos quandomenos quandomuito sequer so somente apardisso ademais afinal ainda alem alias como e enao emsuma enfim mastambem muitomenos naoso nem oumesmo porsinal tambem tampouco assim comisso comoconsequencia consequentemente demodoque destemodo emdecorrencia entao logicamente logo nessesentido pois porcausa porconseguinte poressarazao porisso portanto sendoassim ou ouentao oumesmo nem comose deumlado poroutrolado maisque menosque naoso tanto quanto tao como desdeque docontrario emlugar emvez enquanto nocaso quando se seacaso senao decertaforma dessemodo emfuncao enquanto issoe jaque namedidaque nessadirecao nointuito nomesmosentido ouseja pois porque que umavezque tantoque vistoque aindaque aocontrario apesarde contrariamente contudo embora entretanto foraisso mas mesmoque naoobstante naofosseisso noentanto paratanto pelocontrario porsuavez porem postoque todavia"
    modalizacao = "achar aconselhar acreditar aparente basico bastar certo claro conveniente crer dever dificil duvida efetivo esperar evidente exato facultativo falar fato fundamental imaginar importante indubitavel inegavel justo limitar logico natural necessario negar obrigatorio obvio parecer pensar poder possivel precisar predominar presumir procurar provavel puder real recomendar seguro supor talvez tem tendo ter tinha tive verdade decidir"
    valoracao = "absoluto algum alto amplo aproximado bastante bem bom categorico cerca completo comum consideravel constante definitivo demais elevado enorme escasso especial estrito eventual exagero excelente excessivo exclusivo expresso extremo feliz franco franqueza frequente generalizado geral grande imenso incrivel lamentavel leve maioria mais mal melhor menos mero minimo minoria muito normal ocasional otimo particular pena pequeno pesar pior pleno pobre pouco pouquissimo praticamente prazer preciso preferir principal quase raro razoavel relativo rico rigor sempre significativo simples tanto tao tipico total tremenda usual valer"
    sentimento = "abalar abater aborrecer acalmar acovardar admirar adorar afligir agitar alarmar alegrar alucinar amar ambicionar amedrontar amolar animar apavorar apaziguar apoquentar aporrinhar apreciar aquietar arrepender assombrar assustar atazanar atemorizar aterrorizar aticar atordoar atormentar aturdir azucrinar chatear chocar cobicar comover confortar confundir consolar constranger contemplar contentar contrariar conturbar curtir debilitar decepcionar depreciar deprimir desapontar descontentar descontrolar desejar desencantar desencorajar desesperar desestimular desfrutar desgostar desiludir desinteressar deslumbrar desorientar desprezar detestar distrair emocionar empolgar enamorar encantar encorajar endividar enervar enfeiticar enfurecer enganar enraivecer entediar entreter entristecer entusiasmar envergonhar escandalizar espantar estimar estimular estranhar exaltar exasperar excitar execrar fascinar frustar gostar gozar grilar hostilizar idolatrar iludir importunar impressionar incomodar indignar inibir inquietar intimidar intrigar irar irritar lamentar lastimar louvar magoar malquerer maravilhar melindrar menosprezar obcecar odiar ofender pasmar perdoar preocupar prezar querer recalcar recear reconfortar rejeitar repelir reprimir repudiar respeitar reverenciar revoltar seduzir sensibilizar serenar simpatizar sossegar subestimar sublimar superestimar surpreender temer tolerar tranquilizar transtornar traumatizar venerar"
    pressuposicao = "adivinhar admitir agora aguentar ainda antes atentar atual aturar comecar compreender conseguir constatar continuar corrigir deixar demonstrar descobrir desculpar desde desvendar detectar entender enxergar esclarecer escutar esquecer gabar ignorar iniciar interromper ja lembrar momento notar observar olhar ouvir parar perceber perder pressentir prever reconhecer recordar reparar retirar revelar saber sentir tolerar tratar ver verificar"
    lexicons = [outrage, vagueza, argumentacao, modalizacao, valoracao, sentimento, pressuposicao]
    return lexicons


def read_propositions():
    path = "./src/data/fake_news.csv"
    df = pd.read_csv(path)
    texts = df['completo'].tolist()
    return texts


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, default="./src/data/glove_s300.txt")
    parser.add_argument('--input_csv', type=str, default="./src/data/classificadas.csv")
    parser.add_argument('--binary_embedding', default=False, action='store_true')
    parser.add_argument('--is_refine_short', default=False, action='store_true')
    parser.add_argument('--is_refine_long', default=False, action='store_true')
    parser.add_argument('--short_word_limit', type=int, default=100)
    parser.add_argument('--long_word_limit', type=int, default=1000)
    parser.add_argument('--topic_num', type=int, default=5)
    parser.add_argument('--out', type=str, default='output.csv')
    args = parser.parse_args()

    embedding_path = args.embedding_path
    is_binary = args.binary_embedding
    is_refine_short = args.is_refine_short
    is_refine_long = args.is_refine_long
    short_word_limit = args.short_word_limit
    long_word_limit = args.long_word_limit
    topic_num = args.topic_num

    # read short_docs and long docs
    # raw_short_docs = extract_texts(input_csv)
    # raw_long_docs = read_lexicons(lexicon_file)

    print("reading data...")
    raw_short_docs = extract_texts(args.input_csv)
    print(len(raw_short_docs))
    # raw_long_docs = read_lexicons()
    raw_long_docs = read_propositions()

    
    print("mapping...")
    score_matrix = mapping(embedding_path, raw_short_docs, raw_long_docs, topic_num, \
            is_binary, is_refine_short, is_refine_long, short_word_limit, long_word_limit)
    
    print(score_matrix.shape)
    # with open('hidden_topics_random2.csv','w') as out:
    #     out.write('text,out,vag,arg,mod,val,sent,pres\n')
    #     for i in range(len(raw_short_docs)):
    #         out.write('\"'+raw_short_docs[i]+"\","+str(score_matrix[i][0])+','+str(score_matrix[i][1])+','+str(score_matrix[i][2])+','+str(score_matrix[i][3])+','+str(score_matrix[i][4])+','+str(score_matrix[i][5])+','+str(score_matrix[i][6])+'\n')
    scm = pd.DataFrame(score_matrix, columns=["PL_2630"])
    df = pd.read_csv(args.input_csv)
    df = pd.concat([df, scm], axis=1)

    df.to_csv("ht_tweets_prop_wiki-glove-s300_fake_news.csv",index=False)
    
    
    
