# discursos-twitter

In this work, we provide a linguistic based approach to analyze the Brazilian Congress members' discourse on social media. We use a lexicon-based approach to generate features, and provide model explanations using SHAP (Lundberg and Lee, 2017).

We considered different linguistic aspects in our analysis, namely:  argumentation, presupposition, modalization, sentiment, valuation, vagueness, and outrage.  For each linguistic aspect we introduced the corresponding lexicon for Portuguese, and with Words Mover’s Distance (Kusner et al.,2015) we could expand the size of our lexicon, as demonstrated by (Amorim et al., 2018) and (Caio L.M. Jeronimo, 2019) it is a effective method to work expand lexicons. Our main idea is to employ these lexicons as features to represent declarations posted by the members of the Brazilian Congress, so that we can learn machine learning models to predict the popularity of a tweet.

The WMD allow us to access the similarity between two sentences in a meaningful way, even when they have no specific words in common. The main idea behind the algorithm is that it uses word vectors to find the minimum "travel distance" between a declaration and a lexicon.

<img src="https://vene.ro/images/wmd-obama.png" alt="drawing" width="650"/>

 To investigate model decisions, we use SHAP summary plots, which allows us (i) to summarize the importance of a feature, and (ii) to associate low/high feature values to an increase/decrease in output values, through color-coded violin plots built from all predictions. We will further see that different members of parliament have a different summary plot, which shows different strategies between members to engage the public.

### Lexicons

The choice of certain words in discourse defines the discourse of a subject, thereby the bias and intentions can be tracked by lexicons. The theory of argumentation and pragmatics assume implicit markers of positioning in language as clues to the speakers subjectivity. Thus, we developed a list of seven lexicons for Portuguese: 

**Argumentation:** markers of argumentative discourse,including lexical expressions and connectives.

**Presupposition:** markers that suggest the assumption that something is true. 

**Modalization:** expressions that indicates that the writer exhibits a stance towards its statement.

**Sentiment:** includes markers that indicate a state of mind, emotion or a sentiment of the speaker. 

**Valuation:** assigns a value to the facts.

**Vagueness:** includes words and expressions that could indicate vague claims in the discourse.

**Outrage:** includes markers of insult, offense and indignation. 
		
## Requirements
```
pip3 install -r requirements.txt
```
## How to run the code ? 
```
python3 wmd.py --dataset YOUR_DATASET.csv --col TEXT_COLOMN_NAME --emb YOUR_WORD_EMBEDDING --bin 0 --out OUTPUT_FILE.csv

```
- `--dataset` : Dataset in csv format
	
- `--col`	 : Column name containing the text to apply the wmd for feature extraction.
	- Default : `text`
- `--emb`	: Word embedding file loadable via gensim\'s KeyedVectors.

- `--bin`	: Embedding file is in binary format or not.
	- Options : `0` or `1`
- `--out`	: Output file name
	- Default : `wmd.csv`

## Our data

* [Tweets of the National Congress of Brazil members, 2019-2020](https://drive.google.com/file/d/1Z1QQAbdtcXX-4j2v3p6BKWmCX1AZ7KuC/view?usp=sharing)
* [Generated wmd from the file above](https://drive.google.com/file/d/1E5GIS6T4rmr8uwoiIRbqyXqCkLxDw8Ai/view?usp=sharing)
    
## References

E. Amorim, M. Cançado, and A. Veloso.  2018.  Automated essay scoring in the presence of biased ratings.  In *Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT*, pages 229–237

Caio L. M. Jeronimo, Claudio E. C. Campelo, Leandro Balby Marinho, Allan Sales, Adriano Veloso, and Roberta Viola. 2019. Computing with subjectivity lexicons.*LREC*.

S. Lundberg and S. Lee.  2017.  A unified approach to interpreting model predictions.  In *Annual Conference on Neural Information Processing Systems, Neurips, *pages 4765–4774.

M. Kusner,  Y. Sun,  N. Kolkin,  and K. Weinberger.   2015.   From word embeddings to document distances.   In *International Conference on Machine Learning, ICML,* pages 957–966.
