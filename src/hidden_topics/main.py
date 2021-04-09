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
from gensim.models import KeyedVectors
import argparse
import re
import unicodedata

def vectorizeShortDoc(raw_docs, word_vectors, is_refine=False, word_limit=100):
    """
    word vectors for each short doc
    """
    # tokenize
    print("vectorize short docs...")
    docs = []
    for raw_doc in raw_docs:
        docs.append(preprocess.tokenizeText(raw_doc))
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
    w = re.sub(r"[^a-zA-Z0-9?.!,#$]+", " ", w)

    w = w.rstrip().strip()

    w = ' ' + w + ' '
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
    " ou mesmo ":" ou_mesmo ",
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
    " sirva de exemplo ": "sirva_de_exemplo",
    " por exemplo ":"por_exemplo",
    " e se ":" e_se ",
    " ao mesmo tempo ":" ao_mesmo_tempo ",
    " temos de ":" temos_de ",
    " nos calar ":" nos_calar ",
    " bandido bom e bandido morto ":" bandido_bom_e_bandido_morto ",
    " o bem ": " o_bem ",
    " o mal ": " o_mal ",
    " a mentira ":" a_mentira ",
    " o povo ":" o_povo "
}

rep = dict((re.escape(k), v) for k, v in rep.items()) 
pattern = re.compile("|".join(rep.keys()))

def extract_texts(df):
    texts = df['text'].tolist()
    clean_texts = []
    for text in texts:
        txt = preprocess_sentence(text)
        txt = pattern.sub(lambda m: rep[re.escape(m.group(0))], txt)
        clean_texts.append(txt)
    return clean_texts

def read_lexicons():
    outrage = "mamata desinformacao absurdo denunciar compartilhar sanguinario revolta vandalismo desrespeito desordem caos a_hora_vai_chegar enganar engane guerra acabar_com nao_se_importam nunca_se_importam perde inimigo querem ataques virar_o_jogo contra agora bandidagem ladrao vagabundos povo_brasileiro corruptos horror circo safados ordem imprensa_mentirosa nao_querem_que_voce_saiba canalhas querem_nos querem_fazer ma_fe desmascarado ratos esgoto reagir calar lixo escoria mentirosa corja roubalheira porca inadimissivel inaceitavel massa_de_manobra nao_da_pra_acreditar nos_calar imunda pilantras bandido_bom_e_bandido_morto merda bosta bandidos"
    vagueza = "coisa alguns gente frequentemente provavelmente algumas_pessoas diz_se frequentemente provavelmente bem_comum sustentavel integrado acho claramente dito noticiado talvez especialistas considerado acredita_se bandidos pouco muito orgao_responsavel diversos sempre nunca muito_menos jamais autoridades responsaveis varias mundo_inteiro outros acham bastante propria questao_de_seguranca bom_senso teria urgente classe mestre informacoes situacao  dificil facil demonstrou corruptos local velha_midia velha_politica podridao_vermelha iria viria faria homem mulher coisas_erradas novo_pais nova_nacao isso_dai politica_que_esta_ai questao vicios  politicos elemento nos eles nada verdade inverdade mentira quem em_torno_de ninguem indio que_mais que_menos mal bem vitimizacao nenhum algum tudo o_bem o_mal a_mentira a_verdade o_povo"
    argumentacao = "incluindo inclusive mesmo a_ponto ao_menos apenas ate ate_mesmo incluindo inclusive mesmo nao_mais_que nem_mesmo no_minimo o_unico a_unica pelo_menos quando_menos quando_muito sequer so somente a_par_disso ademais afinal ainda alem alias como e e_nao em_suma enfim mas_tambem muito_menos nao_so nem ou_mesmo por_sinal tambem tampouco assim com_isso como_consequencia consequentemente de_modo_que deste_modo em_decorrencia entao logicamente logo nesse_sentido por_causa por_conseguinte por_essa_razao por_isso portanto sendo_assim ou ou_entao ou_mesmo nem como_se de_um_lado por_outro_lado mais_que menos_que nao_so tanto quanto tao como desde_que do_contrario em_lugar em_vez enquanto no_caso quando se se_acaso senao de_certa_forma desse_modo em_funcao enquanto isso_e ja_que na_medida_que nessa_direcao no_intuito no_mesmo_sentido ou_seja pois porque que uma_vez_que tanto_que visto_que ainda_que ao_contrario apesar_de contrariamente contudo embora entretanto fora_isso mas mesmo_que nao_obstante nao_fosse_isso no_entanto para_tanto pelo_contrario por_sua_vez porem posto_que todavia sirva_de_exemplo por_exemplo e_se ao_mesmo_tempo temos_de temos_que"
    modalizacao = "achar aconselhar acreditar aparente basico bastar certo claro conveniente crer dever dificil duvida efetivo esperar evidente exato facultativo falar fato fundamental imaginar importante indubitavel inegavel justo limitar logico natural necessario negar obrigatorio obvio parecer pensar poder possivel precisar predominar presumir procurar provavel puder real recomendar seguro supor talvez tem tendo ter tinha tive verdade decidir"
    valoracao = "absoluto algum alto amplo aproximado bastante bem bom categorico cerca completo comum consideravel constante definitivo demais elevado enorme escasso especial estrito eventual exagero excelente excessivo exclusivo expresso extremo feliz franco franqueza frequente generalizado geral grande imenso incrivel lamentavel leve maioria mais mal melhor menos mero minimo minoria muito normal ocasional otimo particular pena pequeno pesar pior pleno pobre pouco pouquissimo praticamente prazer preciso preferir principal quase raro razoavel relativo rico rigor sempre significativo simples tanto tao tipico total tremenda usual valer"
    sentimento = "abalar abater aborrecer acalmar acovardar admirar adorar afligir agitar alarmar alegrar alucinar amar ambicionar amedrontar amolar animar apavorar apaziguar apoquentar aporrinhar apreciar aquietar arrepender assombrar assustar atazanar atemorizar aterrorizar aticar atordoar atormentar aturdir azucrinar chatear chocar cobicar comover confortar confundir consolar constranger contemplar contentar contrariar conturbar curtir debilitar decepcionar depreciar deprimir desapontar descontentar descontrolar desejar desencantar desencorajar desesperar desestimular desfrutar desgostar desiludir desinteressar deslumbrar desorientar desprezar detestar distrair emocionar empolgar enamorar encantar encorajar endividar enervar enfeiticar enfurecer enganar enraivecer entediar entreter entristecer entusiasmar envergonhar escandalizar espantar estimar estimular estranhar exaltar exasperar excitar execrar fascinar frustar gostar gozar grilar hostilizar idolatrar iludir importunar impressionar incomodar indignar inibir inquietar intimidar intrigar irar irritar lamentar lastimar louvar magoar malquerer maravilhar melindrar menosprezar obcecar odiar ofender pasmar perdoar preocupar prezar querer recalcar recear reconfortar rejeitar repelir reprimir repudiar respeitar reverenciar revoltar seduzir sensibilizar serenar simpatizar sossegar subestimar sublimar superestimar surpreender temer tolerar tranquilizar transtornar traumatizar venerar"
    pressuposicao = "adivinhar admitir agora aguentar ainda antes atentar atual aturar comecar compreender conseguir constatar continuar corrigir deixar demonstrar descobrir desculpar desde desvendar detectar entender enxergar esclarecer escutar esquecer gabar ignorar iniciar interromper ja lembrar momento notar observar olhar ouvir parar perceber perder pressentir prever reconhecer recordar reparar retirar revelar saber sentir tolerar tratar ver verificar"
    deiticos1 = "eu meu mim tu voce voces"
    #deiticos2 = "nos nosso nossa"
    deiticos3 = "eles deles elas delas neles nelas aqueles"
    #ambiental = "animais animal meio ambiente maus tratos educacao ambiental unidades conservacao animais domesticos proteger protejam problemas sociais biodegradavel parques alimentos areas risco vida barragens caes cao gato publica residuos rejeitos silvestres aguas decomposicao agrotoxicos bioma biomas humanitario equilibrio exoticos lencois freaticos vegetacao assentamento assentamentos mata flora floresta exotica exoticas exotico exoticos hidrologico florestas selva selvagens selvagem indigenas amazonia amazonica paisagens paisagem pequenos produtores pequenos agricultores pequenos proprietarios pantaneira crimes ambientais"
    #agro = "licenciamento ambiental regularizacao ambiental agro agronegocio  agrossilvipastoris atividades agricolas interesse economica economico economicos proprietarios onus produtores cultivo reservatorios artificiais licenca ambiental licenciamento ambiental possuidores perene soberania alimentar cultivo socio organizado desenvolvimento incentivos produtivas agricola empreendedor estrategico excedente financeiramente financeiro legalidade riqueza agropecuaria pecuaria artificial compensatorias compensatoria fomento turismo turistico turisticos racional regulamentacao servidao ambiental sobrevivencia solucionar subsidiar administrativas multifuncionalidade"
    #ambiental2 = "ambiental ambientais socioambiental ambiente ambientalista antiambiental abastece amazonia abate ambiente desmatamento desmonte salles indigenas destruicao preservacao protecao luta meioambiente sustentabilidade povos ambientalistas terras licenciamento defesa devastacao forasalles ibama queimadas  povosindigenas desastre greenpeacebr retrocessos floresta agricultura acordodeparis pantanal mataatlantica caatinga bioma biodiversidade"
    ambiental3 = "ambiental ambientais socioambiental ambiente ambientalista ambientalistas antiambiental amazonia abate desmatamento salles indigenas destruicao preservacao preserva preservar meioambiente sustentabilidade povos terras devastacao #forasalles ibama queimadas  povos_indigenas greenpeacebr floresta florestas agricultura acordo_de_paris pantanal atlantica caatinga bioma biodiversidade animais animal conservacao proteger protejam problemas_sociais biodegradavel parques areas_risco barragens residuos rejeitos silvestres aguas decomposicao agrotoxicos bioma biomas exoticos lencois freaticos vegetacao mar mares oceanos assentamento assentamentos flora exotica exoticas exotico exoticos hidrologico selva selvagens selvagem indigenas amazonia amazonica paisagens paisagem pequenos_produtores agricultores pantaneira crimes_ambientais matas licenciamento_ambiental regularizacao_ambiental agro agronegocio  agrossilvipastoris cultivo reservatorios licenca_ambiental agricultores_familiares agricultura_familiar soberania_alimentar cultivo socio_organizado desenvolvimento incentivos terras_produtivas agricola agricolas agropecuaria pecuaria ecoturismo natural naturais sustentável ecologia ecologico renovavel limpa mst caboclos caboclo quilombos quilombolas wwf reforma_agraria assentamentos ong ongs cidades_verdes fogo ibama icmbio arvore arvores plantas reserva madeira nativa nativo nativas nativos queimadanao operacaoverdebrasil2 cerrado chamas"
    governo = "governo_federal congresso reformas  reforma  guedes mourão crescimento_do_brasil base_do_governo coalisao ministro ministerio teto_de_gastos renda_cidada reforma_tributaria reforma_administrativa tributaria administrativa"
    bolsolavista = "capitao comunismo uma_venezuela lider_da_nacao nacao jairbolsonaro presidente_bolsonaro jair patriota patriotismo nosso_compromisso ideologico ideologicos ideologia bandeira_nacional povo_brasileiro comunismo comunista comunistas anticomunista socialismo socialistas socialista esquerdista esquerdopata terrorismo terrorista terroristas conservador conservadora conservadores brasil_acima_de_tudo deus_acima_de_todos patria_amada mito pr escoria foro_de_sp o_povo_nao_aguenta_mais antivitimista vitimista vitimistas politicamente_correto missao missõ  es crista cristao cristaos deus ideologia_de_genero familia a_verdade conhecereis_a_verdade_e_a_verdade_vos_libertara globolixo globo_lixo globalismo globalista lider_eleito"
    democracia = "democracia democratico democratica democraticas democraticos"
    rt_auto = "proposta reforma imposto debate comissao economia reformatributaria preciso justa tributo emprego entrevista guedes discutir pec ricos previdencia federal fiscal taxacao"
    rt_auto2 = 'proposta imposto reforma debate economia comissao preciso reformatributaria justa tributacao pauta guedes emprego discutida pec discussao taxacao'
    rt_manual = "reforma tributaria tributo imposto carga arrecadacao icms ipi imposto renda fortunas dividendos progressivo progressiva progressividade regressivo regressiva appy laffer pec45 pec110 cbs cpmf"
    ra = "servidor reforma governo preciso ministerio privilegios gastos congresso procuracao semanal reformaadministrativa proposta centrao coaf corte"
    ra2 = 'servidor governo reforma camara guedes congresso publico tributo coaf privilegio estadual debate incluir enviar reformaadministrativa parlamentares poder gastos'

    rp = 'trabalho aprovacao proposta reforma aposentados voto debate pobre votacao guedes previdencia aprovacao comissao reformadaprevidencia luta aposentadoria privilegios economia reformanao'

    armas = 'arma posse decreto morte projeto acesso violencia seguranca tiro importacao rural cidadao policia crime suzano registro municao liberacao'
    racismo = 'negro quilombola luta crime estrutura combate violencia racista vidasnegrasimportam consultaquilombolaja raca antirracista preconceito existe ostracismo protesto'
    covid = 'covid corona coronavirus combate mortes contra enfrentamento covid19 crise casos testagem vacina vidas medicos virus emergencial paciente isolamento'
    imp = 'impeachment impeachmant pedido forabolsonaro democracia renunciamoro governo impeachmentja renuncia processo golpe crime ditadura todospeloimpeachment renuncia tentativa impeachmentbolsonarourgente impeachmentdebolsonaro militar'

    lexicons = [outrage, vagueza, argumentacao,modalizacao, valoracao, sentimento, pressuposicao, ambiental3, rt_auto2, ra2, rp, armas, racismo, covid, imp]
    #lexicons = [ambiental3,rt_auto2, ra2, rp]
    return lexicons


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, default="../src/data/skip_s300.txt")
    parser.add_argument('--input_csv', type=str, default="../src/data/tweets_parlamentares_clean.csv")
    parser.add_argument('--binary_embedding', default=False, action='store_true')
    parser.add_argument('--is_refine_short', default=False, action='store_true')
    parser.add_argument('--is_refine_long', default=False, action='store_true')
    parser.add_argument('--short_word_limit', type=int, default=300)
    parser.add_argument('--long_word_limit', type=int, default=1000)
    parser.add_argument('--topic_num', type=int, default=10)
    parser.add_argument('--out', type=str, default='output.csv')
    args = parser.parse_args()

    embedding_path = args.embedding_path
    is_binary = args.binary_embedding
    is_refine_short = args.is_refine_short
    is_refine_long = args.is_refine_long
    short_word_limit = args.short_word_limit
    long_word_limit = args.long_word_limit
    topic_num = args.topic_num
    output = args.out


    print("reading data...")
    df = pd.read_csv(args.input_csv,lineterminator='\n').dropna(subset=['text']).reset_index()
    raw_short_docs = extract_texts(df)
 

    #raw_short_docs = read_lexicons()

    print(len(raw_short_docs))
    raw_long_docs = read_lexicons()

    print("mapping...")
    score_matrix = mapping(embedding_path, raw_short_docs, raw_long_docs, topic_num, \
            is_binary, is_refine_short, is_refine_long, short_word_limit, long_word_limit)
    
    #print(score_matrix)
    # with open('hidden_topics_random2.csv','w') as out:
    #     out.write('text,out,vag,arg,mod,val,sent,pres\n')
    #     for i in range(len(raw_short_docs)):
    #         out.write('\"'+raw_short_docs[i]+"\","+str(score_matrix[i][0])+','+str(score_matrix[i][1])+','+str(score_matrix[i][2])+','+str(score_matrix[i][3])+','+str(score_matrix[i][4])+','+str(score_matrix[i][5])+','+str(score_matrix[i][6])+'\n')
    scm = pd.DataFrame(score_matrix, columns=["Outrage", "Vagueness", "Argumentation", "Modalization", "Valuation", "Sentiment", "Presupposition",'Ambiental',"rt_auto",'ra_auto','rp_auto','armas_auto','racismo_auto','covid_auto','impeachment_auto'])
    #scm = pd.DataFrame(score_matrix, columns=["ambiental","rt_auto","ra_auto",'rp_auto'])
    df = pd.concat([df, scm], axis=1)

    df.to_csv(output,index=False)

    #print(scm)
    
