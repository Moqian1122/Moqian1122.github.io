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
from langchain_community.emdeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

# We specifiy an embedding rule first.
embeddings = OpenAIEmbeddings(api_key=api_key, model = "text-embedding-ada-002")

# We embed documents according to the rule and create a destination to save the vector database.
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    path = "/tmp/local_qdrant",
    force_recreate = True
)
```

A local vector database is created. A query could then be pushed into the vector as well and `similarity_search()` will help to find the most useful information.

```python
query = "What technical skills are promising for AI careers?"
found_docs = qdrant.similarity_search(query)

print(found_docs[0].page_content)
```

We have the output below from the above exprimental code:
```
educate potential employers about some elements of your work.

As you go through each step, you should also build a supportive community. Having friends and 

allies who can help you — and who you strive to help — makes the path easier. This is true whether 

you’re taking your first steps or you’ve been on the journey for years.

PAGE 7
CHAPTER 2
Learning Technical 
Skills for a Promising 
AI Career

PAGE 8

LEARNING
Learning Technical Skills For a Promising AI Career

CHAPTER 2
```

It actually finds where I picked the question in the query. While the `similarity_search()` gives more than one chunk of documents, it's nature to realize that the `k` as from the `top-k` similar outcomes could be a hyperparameter to be tuned, as well as `chunk_size`and `chunk_overlap`.

It's a bit challening to follow the guidebook after loading, chunking (splitting) and embedding. AI techniques develop so drastically that old LangChain modules illustrated there are already deprecated. Therefore, I explored the latest 1.0.x LangChain documentation by myself and used the most updated modules to proceed. I follow a [2-step RAG](https://docs.langchain.com/oss/python/langchain/rag#rag-chains) featuring a dynamic prompt which absorbs the retrieved documents as the context.

We need certain modules.

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
from langchain.openai import ChatOpenAI
```

To create an agent, it requires configurations such as model, tools and middlewares.

```python
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    last_query = request["messages"][-1].text
    found_docs = qdrant.similarity_search(last_query, 2)
    found_docs_merge = "\n\n".join(found_doc.page_content for found_doc in found_docs)
    system_message = ("You are a helpful and neaty assistant. You are willing to answer with the given context and tend to use bullet points:"
        f"{found_docs_merge}"
    )
    return system_message

model = ChatOpenAI(model='gpt-5', api_key=api_key)
tools = []
middleware = [prompt_with_context]

agent = create_agent(model=model, tools=tools, middleware=middleware)
```

```python
query = "What are the key steps to start a career in AI?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values"
):
    step["messages"][-1].pretty_print()
```

It gives the nice answer from the career guidebook:
```
================================ Human Message =================================

What are the key steps to start a career in AI?
================================== Ai Message ==================================

- Learning (foundational skills)
  - Program in Python; use NumPy, pandas, scikit-learn; basics of PyTorch/TensorFlow
  - Math essentials: linear algebra, probability, optimization; model evaluation and bias/variance
  - Data skills: cleaning, feature engineering, visualization; basic SQL; using notebooks and Git
  - Compute/tooling: GPUs, cloud basics, environments; reading papers and docs effectively

- Projects (build depth, portfolio, and impact)
  - Do end-to-end projects: problem framing → data → baseline → iterate → evaluate
  - Ship something visible: GitHub repos, write-ups, small demos or apps
  - Pick domains you care about; quantify results (accuracy, latency, cost saved)
  - Contribute to open source, join challenges (e.g., Kaggle), seek feedback and code reviews

- Job (turn portfolio into opportunities)
  - Target a role (ML Engineer, Data Scientist, Applied Scientist, AI Product) and align projects
  - Tailor resume/LinkedIn with impact metrics; link repos, demos, and write-ups
  - Network: communities, meetups, OSS contributors, alumni; ask for referrals
  - Prepare interviews: coding, ML fundamentals, system/design, case studies, behavioral
  - Consider internships, apprenticeships, or contract roles as on-ramps

Remember: an important part of any journey is to take the first step, and that step can be a small one. Today, pick one course to start, set up your dev environment, and choose a simple project to build this month.
```

However, the '2-step RAG' has some drawbacks:
- Rigidity: it always go from retrieval to generation. Even it's simply that the user asks 'hey, I just introduce myself to you. Do you remember my name?' The model will still look inside document piles to return you an answer and the answer is wrong because your name is not even in the document.
- Single-turn: it only allows for a single pass and not suitable for multi-turn chats.

With that said, a 2-step RAG buys low latency at expense of memories. Let's now try agentic RAG.

```python
query = "What are the key steps to start a career in AI?"
for step in agent_second.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values"
):
    step["messages"][-1].pretty_print()
```

The outcome is:
```
================================ Human Message =================================

What are the key steps to start a career in AI?
================================== Ai Message ==================================
Tool Calls:
  retrieve_context (call_xzZ1Akr3fugyq4AU1H6nxCNN)
 Call ID: call_xzZ1Akr3fugyq4AU1H6nxCNN
  Args:
    query: key steps to start a career in AI, roadmap, skills, education, portfolio, networking, internships
================================= Tool Message =================================
Name: retrieve_context

[Document(metadata={'producer': 'Adobe PDF Library 17.0', 'creator': 'Adobe InDesign 18.0 (Macintosh)', 'creationdate': '2022-12-13T16:08:00-05:00', 'moddate': '2022-12-13T16:08:04-05:00', 'trapped': 'False', 'total_pages': 41, 'source': '/home/moqian/learning/learn_aiagent/eBook-How-to-Build-a-Career-in-AI.pdf', '_id': '7ca0f335e33146038522027890767f40', '_collection_name': '44f474ba92294e21963c933870be8cb0'}, page_content='companies large and small, navigate careers in AI.\n\nHere’s a framework for charting your own course.\n\nThree  key  steps  of  career  growth  are  learning  foundational  skills,  working  on  projects  (to \n\ndeepen your skills, build a portfolio, and create impact), and finding a job. These steps stack \n\non top of each other:\n\nLEARNING\n\nPROJECTS\n\nJOB\n\nInitially, you \n\nAfter having gained \n\nLater, you will work \n\nfocus on learning \n\nfoundational technical \n\non finding a job.'), Document(metadata={'producer': 'Adobe PDF Library 17.0', 'creator': 'Adobe InDesign 18.0 (Macintosh)', 'creationdate': '2022-12-13T16:08:00-05:00', 'moddate': '2022-12-13T16:08:04-05:00', 'trapped': 'False', 'total_pages': 41, 'source': '/home/moqian/learning/learn_aiagent/eBook-How-to-Build-a-Career-in-AI.pdf', '_id': 'd4e05b62cfe54b5d925a29a8fb7dcecf', '_collection_name': '44f474ba92294e21963c933870be8cb0'}, page_content='important part of any journey is to take the first step, and that step can be a small one.\n\nPAGE 33\n\x0cCHAPTER 10\nKeys to Building a \nCareer in AI\n\nPAGE 34\n\nJOBS\n\x0cKeys to Building a Career in AI\n\nCHAPTER 10\n\nThe path to career success in AI is more complex than what I can  cover in one short eBook. \n\nHopefully the previous chapters will give you momentum to move forward. \n\nHere are additional things to think about as you plot your path to success:')]
================================== Ai Message ==================================

A simple, proven roadmap is: learn foundational skills → build projects → get a job.

1) Learn foundational skills
- Programming: Python, NumPy/Pandas, scikit-learn; SQL; Git; basic Linux. 
- Math: linear algebra, calculus (basics), probability/statistics.
- ML/AI basics: supervised/unsupervised learning, evaluation metrics, overfitting/regularization.
- Deep learning: neural nets, CNN/RNN/Transformers, PyTorch or TensorFlow; LLM fundamentals if relevant.
- Data/MLOps: data cleaning, feature engineering, experiment tracking, cloud basics (AWS/GCP/Azure), deployment concepts; AI ethics.

2) Build projects to deepen skills, create impact, and assemble a portfolio
- Do end-to-end projects on real data (collection → cleaning → modeling → evaluation → simple deployment).
- Reproduce a baseline or paper; participate in Kaggle or open challenges.
- Ship something: a small web app/API/notebook with clear README, visuals, and results.
- Document and share: GitHub, brief blog posts, demos; quantify impact (accuracy lift, latency cut, cost saved).
- Collaborate: contribute to open source or team projects to learn code review and tooling.

3) Find a job
- Target roles that fit your strengths (ML Engineer, Data Scientist, Applied Scientist, MLE Ops/Platform, Analytics).
- Tailor resume/LinkedIn with measurable outcomes; highlight 2–4 strongest projects.
- Network: meetups, online communities, alumni; ask for referrals and informational chats.
- Apply in batches; consider internships, apprenticeships, or adjacent roles (data eng/analytics) as entry ramps.
- Prepare interviews: coding (Python + data structures), ML theory, practical debugging, case studies, and ML system design.

Starter 30-day plan
- Week 1: Refresh Python, NumPy/Pandas, Git/SQL; pick a domain you care about.
- Week 2: Complete a core ML course module and implement 2–3 algorithms from scratch.
- Week 3: Build a small end-to-end project and push to GitHub with a clear README.
- Week 4: Deploy a simple demo, write a short post, update resume/LinkedIn, and start outreach.

Remember: the most important part is taking the first step—start small, ship early, iterate, and your skills and opportunities will compound.
```
Let's change a question and check:

```python
query = "Can you introduce yourself?"
for step in agent_second.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values"
):
    step["messages"][-1].pretty_print()
```
The outcome is:

```
================================ Human Message =================================

Can you introduce yourself?
================================== Ai Message ==================================

Hi! I’m your AI assistant. I can help with research, writing, coding, brainstorming, data analysis, and career guidance—especially in AI and tech. Tell me what you’re working on, and we’ll tackle it together.
```

It's easy to find out that the second response did not invole a tool calling. Actually, this is the main difference. 2-step RAG applies 'middleware' to inject the relevant documents into the prompt. Agentic RAG has a fixed system prompt. The retrieval is a tool and assigned to the 'tools' argument. An agentic RAG could use the agent's discretion to decide whether to call the tool or not based on the query.

To conclude it, the standard flow is loading, chunking and embedding. When it comes to retrieval, several solutions like 2-step RAG and agentic RAG are possible.