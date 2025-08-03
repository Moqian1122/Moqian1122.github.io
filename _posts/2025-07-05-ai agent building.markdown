---
layout: post
title:  "Reviving Proust: a hands-on journey of building an iconic toy AI-agent"
date:   2025-07-05 15:55:01 +0100
categories: jekyll update
---

AI-agent is more useful in real production environments compared to general LLMs. Firstly, the customized external function calling makes AI-agents more specific to its purpose. Secondly, thanks to mechanisms like Retrieval Augmented Generation (RAG), AI-agents' knowledge is more specilised in a certain domain or a certain orgnization. Building specialised AI-agents based on open-sourced LLMs, therefore, is becoming a productive practice in firms and companies. LangChain is a professional and popular framework to build AI-agents leveraging APIs of mainstream LLMs (such as OpenAI, Gemini and DeepSeek), equipped with well-filed documentations and easily-understood prototyping workflow.

The goal of this post is to show how I built a toy AI-agent called 'Talk with Proust', which speaks in the narrative of Marcel Proust. The agent strictly follows the 'chain' building logic as it's an output of learning. Initially, a consideration is given to RAG integration to this AI-agent. However, due to the works of Marcel Proust must have been used in the training of LLMs, chances are that the outcomes from an AI-agent with RAG and one without RAG would be similar. So, I do not put RAG in this post and leave it to future expriments.

Marcel Proust (10 July 1871 – 18 November 1922) is one of the greatest French writers. His idiocyncratic writing style features a continuous, comforting murmuring where time goes in a non-linear way. The non-linearity, even though in conflicts with a forever forward-moving physical world, amazingly unveils humans' transcendental experience about time and memories. In Proust's world of rhetorics, time is a running stream while pieces of memories are swirls randomly floating on the surface. Therefore, it's nice to have an independent AI-agent to talk like Proust, talking with whom would give some literary aesthetic experience to users or Proust's fans.

In this development, we will use OpenAI's o3-mini. To use any external LLM, its API key is required to be subscribed first. Once subscribed, a key should be put in the environment file '.env' in the same working directory. Gradio provides a framework combining frontend and backend. So we will develop an AI-agent chain with LangChain and embed it with Gradio.

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
</pre>

By the code below, we are able to read the OpenAI API key from the environment variable.

<pre>
python

import os
from dotenv import load_dotenv
load_dotenv(override=True)

OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")
</pre>

Below is the most substantial part of the code file. Based on the principle of OOP, we would like to write a function so that it could be called in the Gradio framework. Any initialization of a chatbot requires three basic components: prompt, LLM, and parser. The chain could be expressed as 'promopt \| model \| parser'. The prompt serves as a background setting. Via 'ChatPromptTemplate' I tell the model what kind of a role and what output style this AI-agent should have. This is key as it shapes the unique style of the AI-agent. It should also provide instructions on how to understand the potential user input. With a 'placeholder', it's able to 'remember' the previous conversation so that it could reuse the information given before to give more related response.

The parser is to parse the output of the LLM into the form expected by the related users. For example, if an user expected a more structured output, we might want a JSON. We might also consider multiple chains. If we add another LLM model right after the JSON formot, it might already give some explanations about the result and even print it to a PDF file. That's how chain mechanisms are stronger.

By using the 'async def', it would return anything that is already out of the model instead of returning everything all at once. This is crucial when we want a smooth conversation flow effect.

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

The Gradio block is responsible for layout controlling. CSS is essential here as it's about how the application would look like.

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

Now the codes are finalized. With launch() method we are able to see the prototpye on premise. The prototype is shown as below.

![Initial state of 'Talk with Proust'](/assets/images/talk_with_proust_screenshot.png)

Let's input a question and the dialogue result could be refered as below.

![An example result from Talk with Proust'](/assets/images/talk_with_proust_result.png)

Now here we go. Let's talk with Marcel Proust!
