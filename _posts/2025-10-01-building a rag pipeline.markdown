---
layout: post
title:  "Building a RAG pipeline: knowledge check and practice"
date:   2025-10-01 08:23:04 +0200
categories: AI
---

A previous article builds basics of establishing GAI agents. As GAI agents are normally designed for specific tasks, they are stronger if they are highly customized. The customization falls into two categories: fine-tuning and retrieval-augmented generation (RAG). RAG is productive in that it reduces computational overhead and hardware concerns as it then only requires smaller training datasets without hurting the LLM backing it which is good in generation extenstive contents. In this article, I would like to review myself with the fundamental conceptions of RAG and practising by implementing a real RAG pipeline with Python.

## A conceptual GAI agent pipeline with RAG ##

A canonical GAI agent pipeline: prompt and user queries are input to an LLM. LLM gives response. A final textual output is returned to users.

Where an GAI pipeline with RAG differs from a canonical one is that prompt and user queries are on the first hand going through a referenced external sources to retrieve very relevant specific information. Prompt, user queries and retrieved information are altogether inserted back to the input to an LLM. Other processes from a canonical flow hold.

## A popular conceptual RAG pipeline ##

RAG itself, consist of several steps conceptually.

### Step 0: identifying external sources ###

While tutorials and documentation all start at 'step 1', which is about dealing with referenced external sources, I think it's non-trivial to carefully identify which sources play vital rules to certain tasks. If external sources are too broad so that it is close to general Internet information, it will be meaningless to have a RAG. Extra computation resources allocated to overbroad sources lead to waste. A more carefully selected external source system is vital to make RAG really augmenting agent performance.

### Step 1: document chunking ###

In the external source system there would be many documents in a mix of .doc, .pdf, .csv, .txt, etc. Amongst, .doc, .csv, .txt are machine-readable. The main issue is how to handle .pdf. Regardless, documents are chunked and RAG handles chunk by chunk. There are three reasons why documents need to be chunked:

- **Input length limit**: Enterprise-level documents, if not chunked, could be thousands of tokens, exceeding the input limitation as one single input.
- **Relevance and accuracy**: Since the embedded document will be compared the embedded query according to a certain similarity metric, it's better to chunk documents into pieces so that a small piece of very relevant content could be retrieved. After all, we do not expect that the agent still returns a very long document and in the end users still have to read through a long document.

### Step 2: embedding: chunk-level vectorization ###

Document chunks will be embedded into numerical vectors and stored. Typically, one chunk is one vector.

### Step 3: embedding of queries and similarity calculation ###

Based on a certain similarity metric, there would be an quantitative score for each pair of query and chunk. The k chunks from the pairs that get the top k highest scores will be retrieved.

## Implementing a RAG pipeline ##

The implementation follows the instructions from a handbook about RAG [A Practical Approach to Retrieval Augmented Generation Systems](https://mallahyari.github.io/rag-ebook/03_prepare_data.html).

First of all, I decided to use a guidebook from [DeepLearning.AI](https://www.deeplearning.ai/) about how to have a successful career in AI as the external source example. The guidebook is saved as a PDF file located in the working directory where the notebook is in. Now we want to import the PDF file and transform it to machine-readable format. For Python, there are modules like PyPDF2, pdf2txt and PDFMiner available. Additionally, what if inside a PDF is not selective text but scanned images instead? Then we might consider Optical Character Recognition (OCR). I think this will involve computer vision (CV).

To install and use PyPDF2, we do:

```python
!pip install PyPDF2
from PyPDF2 import PdfReader
```

With a defined 'source_path' to the PDF file, we do:

```python
reader = PdfReader(stream=source_path)
```

Then we get PdfReader object. Calling its 'pages' property, we have a list with each element from a page. Therefore, by using PyPDF2 we naturally have the page-level segmentation of a document.

```python
documents = []

for page in pages:
    documents.append(page.extract_text())

print(documents[0])
```

An empty list named as 'documents' is initialized and each page is textualized as a single string and put into the 'documents' list as one element.

However, doing this causes problems. Each page will actually act as one document in the list 'documents'. The page-level segmentation is suboptical as some pages are lengthy while some pages are just short. I printed the first page and the fifth page to check the page length as below. There is indeed a huge gap.

```
The length of the first document is: 110
The length of the sixth document is: 1151
```

Even segmentation is prefered. A more optimal way is to concatenate pages to a whole documents and chunk it evenly with a specified size. For each chunk, we embed/vectorize it and insert it to the database.

With PyPDF2, the pages-page-document-chunk flow requires manual work. While it's easier with frameworks like langchain and llama-index. To load PDF documents to texts, we do:

```python
from langchain_community.document_loaders import PDFMinerLoader

loader = PDFMinerLoader(source_path)
pdf_content = loader.load()
```

`pdf_content[0].metadata` and `pdf_content[0].page_content` give outputs below:

```
{'producer': 'Adobe PDF Library 17.0',
 'creator': 'Adobe InDesign 18.0 (Macintosh)',
 'creationdate': '2022-12-13T16:08:00-05:00',
 'moddate': '2022-12-13T16:08:04-05:00',
 'trapped': 'False',
 'total_pages': 41,
 'source': '/home/moqian/learning/learn_aiagent/eBook-How-to-Build-a-Career-in-AI.pdf'}
```

```
'A Simple Guide\n\nHow to \nBuild\nYour\nCareer\nin AI\n\nCollected Insights\nfrom Andrew Ng (truncated as it's too long)
```

After loading, with the help of PDFMinerLoader from langchain-community, we could already have a long document of the PDF file. Now, let's try to chunk it. Two most used modules to split textual documents are CharacterTextSplitter and RecursiverCharacterTextSplitter. Here we go ahead with the former.

```python
from langchain_text_splitter import CharacterTextSplitter

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

text_splitter = CharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)
docs = text_splitter.split_documents(pdf_content)
```

Now, we have document chunks. Chunks have more or less the same length. It's time to embed/vectorize them. Following the book, to embed/vectorize chunks, I use `langchain_community.embeddings`. To store the embeddings/vectors, I use `Qdrant`.

```python

```

