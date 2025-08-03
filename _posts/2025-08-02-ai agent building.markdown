---
layout: post
title:  "Reviving Proust: a hands-on journey of building an iconic toy AI-agent"
date:   2025-08-02 15:55:01 +0100
categories: jekyll update
---

AI-agent is more useful in real production environments compared to general LLMs. Firstly, the customized external function calling makes AI-agents more specific to its purpose. Secondly, thanks to mechanisms like Retrieval Augmented Generation (RAG), AI-agents' knowledge is more specilised in a certain domain or a certain orgnization. Building specialised AI-agents based on open-sourced LLMs, therefore, is becoming a productive practice in firms and companies. LangChain is a professional and popular framework to build AI-agents leveraging APIs of mainstream LLMs (such as OpenAI, Gemini and DeepSeek), equipped with well-filed documentations and easily-understood prototyping workflow.

The goal of this post is to show how I built a toy AI-agent called 'Talk with Proust', which speaks in the narrative of Marcel Proust. The agent strictly follows the 'chain' building logic as it's an output of learning. Initially, a consideration is given to RAG integration to this AI-agent. However, due to the works of Marcel Proust must have been used in the training of LLMs, chances are that the outcomes from an AI-agent with RAG and one without RAG would be similar. So, I do not put RAG in this post and leave it to future expriments.

Marcel Proust (10 July 1871 – 18 November 1922) is one of the greatest French writers. His idiocyncratic writing style features a continuous, comforting murmuring where time goes in a non-linear way. The non-linearity, even though in conflicts with a forever forward-moving physical world, amazingly unveils humans' transcendental experience about time and memories. In Proust's world of rhetorics, time is a running stream while pieces of memories are swirls randomly floating on the surface. Therefore, it's nice to have an independent AI-agent to talk like Proust, talking with whom would give some literary aesthetic experience to users or Proust's fans.

<pre>
python 

import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv(override=True)
</pre>





<pre>
python 

OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")
</pre>



<pre>
python 

async def chat(message: str, history: dict):

    prompt = ChatPromptTemplate.from_messages([
        ("system","You are a very helpful assistant. Your purpose is to respond to users' input in the narrative of Marcel Proust, the great French writer. However, you respond only in English. You only comprehend the information retrieved from the pdf that users upload."),
        ("ai","Bonjour madame ou monsieur! I am Proust. It's my pleasure to talk with you."),
        ("placeholder","{chat_history}"),
        ("user","{input}"),
        ("placeholder","{agent_scratchpad}")
    ])

    model = init_chat_model(model="openai:o3-mini")

    agent = prompt | model | StrOutputParser()

    response = agent.invoke({"input":message})

    yield response
</pre>



<pre>
python 

with gr.Blocks(css=css) as demo:

    with gr.Column(elem_classes=['main-container']):
        gr.ChatInterface(
            fn=chat, # we need to add the agent calling here
            chatbot = gr.Chatbot(elem_classes=['chat-container']),
            textbox = gr.Textbox(placeholder="Feel free to talk with me"), 
            type='messages',
            examples=["Hello Proust, could you introduce yourself?", 
                    "Hello Proust, could you please describe the theme of your writing to me?", 
                    "I am a fan of your novel, Proust!"], 
            title="Talk with Proust",
            css=".gradio-container {background: url('file=marcel_proust_illustration.png')}"
        )


demo.launch()
</pre>

![Initial state of 'Talk with Proust'](/assets/images/talk_with_proust_screenshot.png)
