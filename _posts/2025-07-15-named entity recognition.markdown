---
layout: post
title:  "A toy example implementing NER using spaCy"
date:   2025-07-15 23:09:01 +0200
categories: jekyll update
---

NLP (Natural Language Processing) is a vital topic in the advancement of AI. Inside this specific domain, NER (Named Entity Recognition) plays an important role in business analysis. NE (Named Entities) refers to what is called 'proper noun' in English. The word, "English", is already an NE as it represents a specific language or a specific ethic group. Of course, my name "Moqian" is an NE for a person. Intuitively, NE appears in the singular way with always an UPPERCASE of the first letter. In one word, NER is to find NEs from the target textual data. In my coursework slightly touching NLP, two mainstream tools, spaCy and NLTK have been introduced. This post uses sapCy to implement some simple tasks about NER. 

First of all, let's set up spaCy. I used a vanilla venv based on Python 3.12.3. I downloaded spaCy to this venv.

<pre>
python 

%pip install spacy
</pre>

Unlike normal Python modules, spaCy, as a natural language processing toolkit, naturally requires pre-defined natural language warehouses as benchmarks for text recognition. The warehouses are categorized by different languages. The most frequently used one is 'en_core_wen_sm'. We should learn its naming convention before we deep dive. 'en' obviously stands for 'English'. 'core' refers to vocabulary, syntax, entities, vectors. 'web' means it's web texts obtained in the form of written texts which are blogs, news and comments. The last position in the name refers to the size. For English model, there are 'sm' (small), 'md' (medium) and 'lg' (large) models. spaCy Official claims that the larger the model is, the better its performance is.

<pre>
python

!python -m spacy download en_core_web_lg
!python -m spacy download en_core_web_md
!python -m spacy download en_core_web_sm
!python -m spacy download zh_core_web_sm
</pre>

Do note that the 'core' part contains vectors. That is to say, if we use the setting by default in spaCy, the input text will only be tokenized and matched in a pre-made vector dictionary, illutrated by the picture below.

![Flow of an simple spaCy workflow](/assets/images/spacy_flow.png)

That is why the 'core' contains not only language properties, but also their pre-made vectorized representations.

Let's get hands dirty now. Using a load method with 'en_core_web_sm' initializes an nlp parser. 'doc' is the returned spaCy 'Doc' object. For all the tokens categorized as 'named entities', one of its methods is to show all the NEs appear in the input text. The NEs are put in a tuple. We could check by printing them with a for loop using 'text' and 'label_' to see what text is obtained and what NE category it falls into. Do not mix up "ent.label" and "ent.label_"! "label" is the ID of the NE category with which spaCy model uses as a key to find the real category value.

<pre>
python 

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startp for $1 biilion.")

for ent in doc.ents:
    print(ent.text, ent.label_)
</pre>

The result is as below. The NEs are out with their NE categories.

<pre>

Apple ORG
U.K. GPE
$1 biilion MONEY
</pre>

Now, let's try another but in Chinese!

<pre>

nlpch = spacy.load("zh_core_web_sm")
textch = "近期，根据路透社新闻的消息，马云已经不再担任阿里巴巴公司的首席执行官。他热心于慈善事业，捐赠5千万人民币给中国以及非洲的偏远小学。"
docch = nlpch(textch)
for ent in docch.ents:
    print(ent.text, ent.label_)
</pre>

Check the result below. The result looks promising except a '偏远小学' (rural elementary school) is not really an NE. Generally, the result is satisfactory.

<pre>
路透社 ORG
阿里巴巴公司 ORG
5千万人民币 MONEY
中国 GPE
非洲 LOC
偏远小学 ORG
</pre>