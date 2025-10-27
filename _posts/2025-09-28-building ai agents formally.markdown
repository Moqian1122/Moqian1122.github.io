---
layout: post
title:  "Building AI agents in details using frameworks"
date:   2025-09-28 11:12:33 +0800
categories: AI
---

This article shows efforts made to build generative AI agents. I follow instructions from the book [Building Generative AI Agents](https://learning.oreilly.com/library/view/building-generative-ai/9798868811340/) by Tom Taulli and Gaurav Deshmukh. The book arranges it contents well in that the first few chapters give taxonomy of LLMs and AI agents from different perspectives and afterwards it starts to get really technical. The article deals with initializing a basic chat with LLM APIs. The official [openai python source](https://github.com/openai/openai-python) is also refered to in this article.

## Why we want to use frameworks ##

While it is nice to have a proprietary language model deployed, there are multiple reasons for enterprises to go for LLMs.

- **Cost**: it requires professional technical staff to build the architecture (which would be more or less just another transformer model) and extremely extensive data to train the model, which leads to very large cost to initiate such a system.
- **Hardware**: training on super extensive datasets (not only in volume but also in various formats) requires high-performance hardware such as GPUs.
- **Transfer** learning: transfer learning is a machine learning technique term. It means that a model trained for a certain purpose could be applied to another but relevant task. LLMs, while trained on vast amounts of general data across the Internet, shows the unexpected ability to potentially perform well on specific tasks with a bit of fine-tuning. Given this fact, we do not have to bother to build a system from scratch.

Therefore, in many business cases, we just go with frameworks with fine-tuning or RAG.

## Prequisites: API and key ##

To make use of LLMs, we need the corresponding API. In this article, we go ahead with OpenAI's GPT models.

To get access to GPT models in the script, we need to connect to the OpenAI API. Its pythonic one is 'openai'. In the environment, we do 'pip install openai' to have OpenAI SDK (Software Development Kit) installed.

A client must be initialized everytime it is called and used. To initiate a client, we need an API Key. An API must be requested [here](https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key?language=python). As far as I ever see, there are three ways to load the key into the working environment as an environment variable.

- **Terminal**: try the command line below in the terminal
```bash
export OPENAI_API_KEY="your_api_key_here"
```
- **In script using .env file**: in the same working directory where the script is located in, create a new file named exactly as '.env'. In the '.env' file, write the following text:
```
OPENAI_API_KEY = "your_api_key_here"
```
In the script, do the following:
```python
import os
from load_dotenv import load_dotenv
```
Then write codes as:
```python
load_dotenv(override=True)
openai_key_api = os.getenv('OPENAI_API_KEY')
```
Actually, this line of code will exact match "OPENAI_API_KEY" automaticallty in the ".env" file and add it as the environment variable. The 'load_dotenv(override=True)' means that if an environment variable called 'OPENAI_API_KEY' already exists, then it brute-force changes the environment variable value from the old value to the new value (from an old key to the new key in our case here).

However, when doing this, we shall be cautious that it might leak. If that's the case we have a big bill to pay! So we need to add '.env' to the '.gitignore' file to avoid bankruptcy from exposing the key information to GitHub.

- **In script using getpass**: we make use of the 'getpass' package to receive the input from users. We do the following:
```python
from getpass import getpass

def get_api_key():
    return getpass("Enter your api key: ")
```
In this way, new users are required to use their own API keys everytime they want to start a new session. In this way there is no risk of key leak.

I already listed three ways to load the key. Now, with the key we get closer to magic. The initialization of a client is as below assuming we do 'in script key loading using .env file' thus already having 'openai_api_key':

```python
client = openai.OpenAI(api_key=openai_api_key)
```

## Practice: a complete, discrete chat ##

We follow the book where the first task is to write a tweet generator. While the codes are available there, I would like to just save my own codes here for relfections. While the official already recommends to use 'Responses' with the latest features, we use 'Chat Completions' for now as it is instructed in the book.

The block is simply generating single model response instead of entering a conversation stream. Literally, Chat Completions' just means to complete a chat. Since a system prompt and a user prompt are already given, a response as result will complete the chat. I wrote two blocks. The only difference between the two blocks is the "role". The first completion is "system" while the second completion is "developer". I did this because I was curious to see whether there would be difference in their functionalities.

```python
completion = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = [
        {"role": "system", "content": "You are a social media expert skilled at creating engaging tweets."},
        {"role": "user", "content": "Write a one-sentence tweet about Donald Trump. Include relevant emojis."}]
    )
```

```python
completion_second = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = [
        {"role": "developer", "content": "You are a social media expert skilled at creating engaging tweets."},
        {"role": "user", "content": "Write a one-sentence tweet about Donald Trump. Include relevant emojis."}]
    )
```

Now check two outcomes. First of all, I checked the type of the completion. It is 'openai.types.chat.chat_completion.ChatCompletion'. Printing 'completion', we get:
```
ChatCompletion(id='chatcmpl-CUlE2maAOWpXeH4CIzN5bUw9jbu1D', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='🌍🇺🇸 Donald Trump continues to stir the political arena, leaving supporters and critics alike buzzing with his latest moves! 🤔⚡️ #Politics #Trump2024', refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None))], created=1761447410, model='gpt-4o-mini-2024-07-18', object='chat.completion', service_tier='default', system_fingerprint='fp_560af6e559', usage=CompletionUsage(completion_tokens=37, prompt_tokens=37, total_tokens=74, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
```

It gives much information about the response. It's easy to note that the textual response is exactly within the outcome. Before we filter it out, we want to also check other information. The 'id' is obviously the unique identifier for this response. 'choices' is a list. It is actually all the responses that the completion gives. As the book mentioned, one difference between traditional software development and AI agent development is that traditional software development gives deterministic outputs while AI agents give probabilitic outputs. Therefore, it's possible for LLMs to give more than one answer to a request. In the 'client.chat.completions.create(**kwrgs)', 'model' and 'messages' are compulsory arguments. Besides, there is an optional argument 'n', specifying the number of outputs that we prefer the LLM to return. 'n' is by default 1. That is why here we will just see one response in 'choices'. For each choice, the core is the 'message' and 'content' in the 'message'. Therefore, I use
```python
print(completion.choices[0].message.content)
```
to get the output below.
```
🌍🇺🇸 Donald Trump continues to stir the political arena, leaving supporters and critics alike buzzing with his latest moves! 🤔⚡️ #Politics #Trump2024
```
I apply the same trick to 'completion_second' and the output shows below:
```
Donald Trump continues to make headlines as he navigates the ever-changing political landscape! 🇺🇸🗞️ #Politics #Trump2024
```
We do see two tweets are not the same. But both make syntax and semantics sense. This validates the stochastic and probabilistic nature of LLM models and their agentic applications. Also, it shows that "developer" or "system" could be used interchangeably.

In the book, the block is wrapped by a function:
```python
def get_api_key():
    return getpass("Please enter your OpenAI API Key: ")

def generate_tweet(client, topic):
    try:
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role": "system", "content": "You are a social media expert skilled at creating engaging tweets."},
                {"role": "user", "content": f"Write a one-sentence tweet about {topic}. Include relevant emojis."}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"An error occurs: {str(e)}"

def main():
    print("Welcome to the Tweet Generator!")

    api_key = get_api_key()
    client = openai.OpenAI(api_key=api_key)

    while True:
        topic = input("Enter a topic for your tweet (or 'quit' to exit): ")
        if topic.lower() == 'quit':
            break
        tweet = generate_tweet(client, topic)
        print("\nGenerated Tweet: ")
        print(tweet)
        print("\n"+"-"*50+"\n")
```
The example codes are more interactive with users. I tested it by giving "topic='Donald Trump'". The output is as followed:
```
Welcome to the Tweet Generator!

Generated Tweet: 
Donald Trump's recent rally stirred up the crowd, fueling debates across the nation! 🇺🇸🔥 #Politics #Trump2024

--------------------------------------------------

```
The code uses a 'while True' loop. The 'while True' is to let users to decide when to quit. So it's often coupled with an 'input'. A 'quit' will be linked to 'break' which immediately stops the loop.

## Practice: a complete, continuous chat ##

As I learned before, a chat with multiple ask and response requires 'memory'. In the latest response, the agent should be able to recall the information given in the previous rounds. How to add memories? In the code from a discrete chat, we just need to try to store the previous information. A possible way is to initialize an empty list immediately whenever a chat completion is created and the latest response can be appended to the list.

```python
def generate_response(client, messages):
    try:
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"An error occurs: {str(e)}"

def main():

    api_key = get_api_key()
    client = openai.OpenAI(api_key=api_key)

    messages = [
                {"role": "system", "content": "You are a social media expert skilled at creating engaging tweets."}
            ]

    while True:
        user_input = input("Please enter anything that you want to ask (type 'quit' if you no longer have questions): ")
        if user_input.lower() == 'quit':
            break
        messages.append({"role": "user", "content": f"{user_input}"})
        response = generate_response(client, messages)
        print(response)
        messages.append({"role": "expert", "content": f"{response}"})
        if len(messages) >= 20:
            print("You have reached the maximum capacity. Please relaunch a new chat.")
            break
```

I tested it and got an error, saying:
```
An error occurs: Error code: 400 - {'error': {'message': "Invalid value: 'expert'. Supported values are: 'system', 'assistant', 'user', 'function', 'tool', and 'developer'.", 'type': 'invalid_request_error', 'param': 'messages[2].role', 'code': 'invalid_value'}}
```
The issue is the role which I take for granted and assign to the messages is not supported. I used 'expert'. But the supported values are 'system', 'assistant', 'user', 'function', 'tool' and 'developer'. That is why in the last section it's valid whether I use 'system' or 'developer'. As long as the agent is created, we should go ahead with 'assistant'. Replacing it, I got the output:

```
Hi there! I'm your go-to social media expert, here to help you craft engaging tweets and enhance your online presence. Whether you need tips on trending topics, witty one-liners, or strategies to boost your engagement, I’ve got you covered! Let’s make your social media shine! 🌟 #SocialMediaExpert
Nice to meet you, Moqian! 🎉 What can I help you with today? Whether it's tweet ideas, engagement strategies, or anything else social media-related, I'm here for you! #LetsGetSocial
Absolutely! Here’s a tweet about Belgium:

"🇧🇪✨ From the stunning architecture of Brussels to the delicious waffles and rich chocolate, Belgium is a hidden gem waiting to be explored! What’s your favorite Belgian treat? 🍫🥨 #TravelTuesday #Belgium" 

Feel free to tweak it to match your style!
Your name is Moqian! How can I assist you further today? 😊
You asked for content under the topic of Belgium. If you have more topics in mind or need additional content, just let me know!
```

The contents are outputs from multiple chat rounds. It indeed 'remembers' my name and my previous request.

The book employs 'Assistant API' which is going to be deprecated in 2026. I teach myself and move forward with Response API. To build multi-turn conversations, the only way in Chat Completions API is to manually append new request and response to a recording list. This was is still valid if we want to migrate to Responses API. An alternative, also a bit smarter way to do it, is using Conversations API together with Responses API.

```python
def generate_response(input):

    response = openai.responses.create(
        model = "gpt-5",
        input = [{"role": "user", "content": f"{input}"}],
        instructions = "You are a very nice and helpful assistant.",
        conversation = conversation.id
    )

    return response.output_text

def main():

    while True:
        user_input = input("Please enter anything that you want to ask (type 'quit' if you no longer have questions): ")
        if user_input.lower() == 'quit':
            break
        print(f"User: {user_input}")
        response = generate_response(user_input)
        print(f"Assistant: {response}")
```

The codes above return:
```
User: Hello! Could u introduce yourself?
Assistant: Hi Moqian! I’m ChatGPT, your AI assistant. I can answer questions, explain concepts, brainstorm ideas, write and edit text, help with math or code, plan projects or trips, and even analyze images. What would you like to do today?
User: Could u tell me what questions I asked you before?
Assistant: Here are the questions you asked earlier in this chat:

- What are the 5 Ds of dodgeball?
- What is my name?
- Could you tell me my name? (asked four times)
- Hello! Could u introduce yourself?
User: You are really good at remembering things. Nice job!
Assistant: Thank you, Moqian! I appreciate it. If there’s anything else you’d like to do or ask, I’m here to help.
```
It does remember my name. This saves extra efforts to manually manage a message history.

Now, it's clear how OpenAI's Chat Completions API (text only) and Responses API (multimodal) could be leveraged in building basic GAI agents, from single-turn conversations to multi-turn conversations.