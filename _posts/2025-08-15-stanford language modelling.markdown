---
layout: post
title:  "Learning with Stanford 02: language modelling from scratch"
date:   2025-08-15 17:32:58 +0200
categories: Artificial Intelligence, LLM
---

This is the second Episode of my 'Learning with Stanford' series. In this Episode, I am following [CS336: Language Modeling from Scratch (Spring 2025)](https://stanford-cs336.github.io/spring2025/). The biggest challenge of the course is 'from scratch'. It covers the whole life cycle from basics such as tokenization, to architecture, to final alignment such as fine-tuning and prompt engineering. The most up-to-date [lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) are available online.

## Lecture notes ##

### Lecture 1: Overview and tokenization ###

In the lecture, there is a tokenization example using a string "Hello, 🌏! 你好!". The compression ratio is calculated as around 1.5385. The compression ratio calculation follows the equation below.

$$
Compression Ratio = \frac{Original Size}{Processed Size}
$$

The problem is that how we get this compression ratio? If a token is basically a character which occupies 1 byte, then shouldn't the ratio be close to 1? The fact is, emojis and non-alphabetic  language characters occupy more than 1 byte. To make very clear of it, let's summarize the rule. A raw text from human language is a UNICODE string of characters. A character from, say, English, occupies 1 byte. Chinese, Korean and Japanese characters occupy 3 bytes per character. An emoji occupies 4 bytes. Common punctuations such as commas, exclamation marks as well as blank spaces take 1 byte. Now, let's count! We tokenize "Hello, 🌏! 你好!" into a numeric vector in which elements represent token indices. The tokens could be written as ['H', 'e', 'l', 'l', 'o', ',', ' ', '🌏', '!', ' ', '你', '好', '!']. Adding up bytes across all original characters and other symbols yields 20 (1+1+1+1+1+1+1+4+1+1+3+3+1=20). The tokenized numeric vector is [72, 101， 108， 108， 111， 44， 32， 127757， 33， 32， 20320， 22909， 33]. The processzed size is down to 13. 20/13 yields about 1.5385.

## Assignments ##

### Assignment 1: Introduction to word vectors ###