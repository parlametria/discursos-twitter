import argparse
import pandas as pd
import numpy as np
import re

from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import euclidean_distances

from pyemd import emd

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words("portuguese"))
stop_words.update(['que', 'até', 'esse', 'de', 'do','essa', 'pro', 'pra', 'oi', 'lá'])

pattern = "(?u)\\b[\\w-]+\\b"

def select_valid_words(text, stop_words=stop_words):
    """
        filter invalid words from the text
    """
    valid = []
    for string in text.split():
        try: #search for embedding vector
            a = W[vocab_dict[string]]
            valid.append(string)
        except:
            continue
    valid = [w for w in valid if not w in stop_words]
    return " ".join(valid)

def calc_wmd(text, lexicon, pattern=pattern ):
    """
        Calculates the Word Mover's Distance between a piece of text and a lexicon, generating a score.
    """
    vect = CountVectorizer(token_pattern=pattern, strip_accents=None).fit([lexicon, text])
    v_1, v_2 = vect.transform([lexicon, text])
    v_1 = v_1.toarray().ravel()
    v_2 = v_2.toarray().ravel()
    W_ = W[[vocab_dict[w] for w in vect.get_feature_names()]]
    D_ = euclidean_distances(W_)
    v_1 = v_1.astype(np.double)
    v_2 = v_2.astype(np.double)
    v_1 /= v_1.sum()
    v_2 /= v_2.sum()
    D_ = D_.astype(np.double)
    D_ /= D_.max()
    distance = emd(v_1, v_2, D_)
    return 1 - distance

def main(args):
    df = pd.read_csv(args.dataset)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    df['clean_tweets'] =  [re.sub(r"(?:\@|https?\://)\S+", '', str(x)) for x in df[args.col]]
    df['clean_tweets'] =  [emoji_pattern.sub(r'', str(x)) for x in df['clean_tweets']]
    df['clean_tweets'] =  [re.sub("(\\d|\\W)+|\w*\d\w*"," ", str(x)) for x in df['clean_tweets']]
    df['clean_tweets'] =  [re.sub(r"\b[a-zA-Z]\b", "", str(x)) for x in df['clean_tweets']]
    df['clean_tweets'] =  [x.lower() for x in df['clean_tweets']]

    outrage = "mamata desinformacao absurdo denunciar compartilhar sanguinario revolta vandalismo desrespeito desordem caos ahoravaichegar enganar engane guerra acabarcom naoseimportam nuncaseimportam perde inimigo querem ataques virarojogo contra agora bandidagem ladrao vagabundos povobrasileiro corruptos horror circo safados ordem imprensamentirosa naoqueremquevocesaiba canalhas queremnos queremfazer mafe desmascarado averdade temosque ratos esgoto reagir calar lixo escoria mentirosa corja roubalheira porca inadimissivel inaceitavel massademanobra naodapraacreditar"
    vagueza = "coisa alguns gente frequentemente provavelmente algumaspessoas dizse frequentemente provavelmente bemcomum sustentavel integrado acho claramente dito noticiado talvez especialistas considerado acreditase bandido pouco muito orgaoresponsavel diversos sempre nunca muitomenos jamais autoridades responsaveis varias mundointeiro outros acham bastante propria questaodeseguranca bomsenso teria urgente classe mestre informacoes situacao  dificil facil demonstrou corruptos local velhamidia velhapolitica podridaovermelha iria viria faria homem mulher coisaserradas novopais novanacao issodai politicaqueestaai questao vicios  politicos elemento nos eles nada verdade inverdade mentira quem emtornode ninguem indio quemais quemenos mal bem vitimizacao nenhum algum"
    argumentacao = "incluindo inclusive mesmo aponto aomenos apenas ate atemesmo incluindo inclusive mesmo naomaisque nemmesmo nominimo ounico aunica pelomenos quandomenos quandomuito sequer so somente apardisso ademais afinal ainda alem alias como e enao emsuma enfim mastambem muitomenos naoso nem oumesmo porsinal tambem tampouco assim comisso comoconsequencia consequentemente demodoque destemodo emdecorrencia entao logicamente logo nessesentido pois porcausa porconseguinte poressarazao porisso portanto sendoassim ou ouentao oumesmo nem comose deumlado poroutrolado maisque menosque naoso tanto quanto tao como desdeque docontrario emlugar emvez enquanto nocaso quando se seacaso senao decertaforma dessemodo emfuncao enquanto issoe jaque namedidaque nessadirecao nointuito nomesmosentido ouseja pois porque que umavezque tantoque vistoque aindaque aocontrario apesarde contrariamente contudo embora entretanto foraisso mas mesmoque naoobstante naofosseisso noentanto paratanto pelocontrario porsuavez porem postoque todavia"
    modalizacao = "achar aconselhar acreditar aparente basico bastar certo claro conveniente crer dever dificil duvida efetivo esperar evidente exato facultativo falar fato fundamental imaginar importante indubitavel inegavel justo limitar logico natural necessario negar obrigatorio obvio parecer pensar poder possivel precisar predominar presumir procurar provavel puder real recomendar seguro supor talvez tem tendo ter tinha tive verdade decidir"
    valoracao = "absoluto algum alto amplo aproximado bastante bem bom categorico cerca completo comum consideravel constante definitivo demais elevado enorme escasso especial estrito eventual exagero excelente excessivo exclusivo expresso extremo feliz franco franqueza frequente generalizado geral grande imenso incrivel lamentavel leve maioria mais mal melhor menos mero minimo minoria muito normal ocasional otimo particular pena pequeno pesar pior pleno pobre pouco pouquissimo praticamente prazer preciso preferir principal quase raro razoavel relativo rico rigor sempre significativo simples tanto tao tipico total tremenda usual valer"
    sentimento = "abalar abater aborrecer acalmar acovardar admirar adorar afligir agitar alarmar alegrar alucinar amar ambicionar amedrontar amolar animar apavorar apaziguar apoquentar aporrinhar apreciar aquietar arrepender assombrar assustar atazanar atemorizar aterrorizar aticar atordoar atormentar aturdir azucrinar chatear chocar cobicar comover confortar confundir consolar constranger contemplar contentar contrariar conturbar curtir debilitar decepcionar depreciar deprimir desapontar descontentar descontrolar desejar desencantar desencorajar desesperar desestimular desfrutar desgostar desiludir desinteressar deslumbrar desorientar desprezar detestar distrair emocionar empolgar enamorar encantar encorajar endividar enervar enfeiticar enfurecer enganar enraivecer entediar entreter entristecer entusiasmar envergonhar escandalizar espantar estimar estimular estranhar exaltar exasperar excitar execrar fascinar frustar gostar gozar grilar hostilizar idolatrar iludir importunar impressionar incomodar indignar inibir inquietar intimidar intrigar irar irritar lamentar lastimar louvar magoar malquerer maravilhar melindrar menosprezar obcecar odiar ofender pasmar perdoar preocupar prezar querer recalcar recear reconfortar rejeitar repelir reprimir repudiar respeitar reverenciar revoltar seduzir sensibilizar serenar simpatizar sossegar subestimar sublimar superestimar surpreender temer tolerar tranquilizar transtornar traumatizar venerar"
    pressuposicao = "adivinhar admitir agora aguentar ainda antes atentar atual aturar comecar compreender conseguir constatar continuar corrigir deixar demonstrar descobrir desculpar desde desvendar detectar entender enxergar esclarecer escutar esquecer gabar ignorar iniciar interromper ja lembrar momento notar observar olhar ouvir parar perceber perder pressentir prever reconhecer recordar reparar retirar revelar saber sentir tolerar tratar ver verificar"
    lexicons = [outrage, vagueza, argumentacao, modalizacao, valoracao, sentimento, pressuposicao]
    lexicons = [select_valid_words(lexicon) for lexicon in lexicons]

    for index,row in df.iterrows():
        text = select_valid_words(row['clean_tweets'])
        df.loc[index,'Outrage'] = calc_wmd(text,lexicons[0])
        df.loc[index,'Vagueness'] = calc_wmd(text,lexicons[1])
        df.loc[index,'Argumentation'] = calc_wmd(text,lexicons[2])
        df.loc[index,'Modalization'] = calc_wmd(text,lexicons[3])
        df.loc[index,'Valuation'] = calc_wmd(text,lexicons[4])
        df.loc[index,'Sentiment'] = calc_wmd(text,lexicons[5])
        df.loc[index,'Pressupposition'] = calc_wmd(text,lexicons[6])
    df.to_csv(args.out,index=False)
    return



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help='csv dataset')
    parser.add_argument("--col", default='text', type=str, help='text column in csv file')
    parser.add_argument("--emb", type=str, help='embedding file which can be loaded via gensim\'s KeyedVectors')
    parser.add_argument("--bin", type=int, help='embedding file is in the binary format or not')
    parser.add_argument("--out", default='wmd.csv',type=str, help='name of generated output')

    args = parser.parse_args()
    args.bin = bool(args.bin)
    wv = KeyedVectors.load_word2vec_format(args.emb, binary=args.bin , unicode_errors="ignore")
    wv.init_sims()
    fp = np.memmap("embed.dat", dtype=np.double, mode='w+', shape=wv.vectors_norm.shape)
    fp[:] = wv.vectors_norm[:]

    with open("embed.vocab", "w") as f:
        for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
            print(w, file=f)
    shape = (len(wv.index2word), wv.vector_size)
    del fp, wv

    W = np.memmap("embed.dat", dtype=np.double, mode="r+", shape=shape)
    with open("embed.vocab") as f:
        vocab_list = map(str.strip, f.readlines())
    vocab_dict={w:k for k, w in enumerate(vocab_list)}

    main(args)