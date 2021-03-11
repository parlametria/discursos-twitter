import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

print("preparing data...")
df = pd.read_csv("../../../data/ht_arm_rac_2019-01-01_to_2021-02-22.csv",lineterminator='\n')#[["nome_eleitoral","partido","created_at","text","ambiental","rt_auto","ra_auto","rp_auto"]]
lexicons = ['Outrage','Vagueness', 'Argumentation', 'Modalization', 'Valuation', 'Sentiment',
       'Presupposition', 'Ambiental', 'rt_auto', 'ra_auto', 'rp_auto',
       'armas_auto', 'racismo_auto']

features = lexicons
df['len_tweet'] = df['text'].str.split().apply(len)
df = df[df["len_tweet"] > 9]
df[features] = (df[features] - df[features].mean()) / df[features].std()
df = df.dropna()

temas = []
for _, row in df.iterrows():
    maxi = max([row.Ambiental, row.rt_auto, row.ra_auto, row.rp_auto, row.armas_auto, row.racismo_auto])
#     if maxi> 1.5:
#         print(maxi)
    if maxi < 1.5:
        temas.append(0)
    elif row.Ambiental == maxi:
        temas.append(1)
    elif row.rt_auto == maxi:
        temas.append(2)
    elif row.ra_auto == maxi:
        temas.append(3)
    elif row.rp_auto == maxi:
        temas.append(4)
    elif row.armas_auto == maxi:
        temas.append(5)
    elif row.racismo_auto == maxi:
        temas.append(6)
df["tema"] = temas

print("generating tsne...")
tsne = TSNE(n_components=2, random_state=0,learning_rate = 350)
X = df[features].values
y = df[["tema"]].values

X_2d = tsne.fit_transform(X)


print("saving tsne...")
X_2df = pd.DataFrame(X_2d)
X_2df.to_csv("tsne_parl_inf.csv",index=False)

target_ids = range(len(features)+1)


print("plotting general tsne")
fig = plt.figure(figsize=(40, 40))

colors =  'grey','g', 'b', 'c', 'm', 'red', 'black'
for i, c, label in zip(target_ids, colors, ["none",'Ambiental','Reforma Tributária','Reforma Administrativa','Reforma da Previdência', "Armas", "Racismo"]):
    plt.scatter(X_2d[np.squeeze(y == i), 0], X_2d[np.squeeze(y == i), 1], c=c, label=label, alpha=0.30)
# for word, (x,y) in zip(nomes, X_2d):
#         plt.text(x+0.005, y+0.005, word,fontsize='xx-small')
plt.legend(markerscale=7,fontsize=30)
#plt.show()
fig.savefig("tsne_temas.png")
