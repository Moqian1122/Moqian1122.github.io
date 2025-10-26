---
layout: post
title:  "Building AI agents in details using frameworks"
date:   2025-09-28 11:12:33 +0800
categories: AI
---

This article shows efforts made to build generative AI agents. I follow instructions from the book [Building Generative AI Agents](https://learning.oreilly.com/library/view/building-generative-ai/9798868811340/) by Tom Taulli and Gaurav Deshmukh. The book arranges it contents well in that the first few chapters give taxonomy of LLMs and AI agents from different perspectives and afterwards it starts to get really technical. The article deals with initializing a basic chat with LLM APIs. The official [openai python source](https://github.com/openai/openai-python) is also refered to in this article.

## Why we want to use frameworks ##

While it is nice to have a proprietary language model deployed, there are multiple reasons for enterprises to go for LLMs.

- Cost: it requires professional technical staff to build the architecture (which would be more or less just another transformer model) and extremely extensive data to train the model, which leads to very large cost to initiate such a system.
- Hardware: training on super extensive datasets (not only in volume but also in various formats) requires high-performance hardware such as GPUs.
- Transfer learning: transfer learning is a machine learning technique term. It means that a model trained for a certain purpose could be applied to another but relevant task. LLMs, while trained on vast amounts of general data across the Internet, shows the unexpected ability to potentially perform well on specific tasks with a bit of fine-tuning. Given this fact, we do not have to bother to build a system from scratch.

Therefore, in many business cases, we just go with frameworks with fine-tuning or RAG.


## Prequisites: API and key ##

## Practice: a complete, discrete chat ##

## Practice: a complete, continuous chat ##

## Practice: retrieval-augmented generation (RAG) ##