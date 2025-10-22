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

### Lecture 2: PyTorch and resource accounting ###

We deal with archeitecture via PyTorch. Before this, we could establish our basic conception of a tensor. Almost all of parameters, optimizer state, data, gradient and activations are stored in the form of tensors.

```python
import torch

torch.zeros(4,8)
```
The code above will generate a 4*\8 matrix fulfilled with 0 s. While dealing with deep learning, we need to be aware of the non-trivial computational efforts and limited resources. One aspect of the resources is the memory. To make most of the memory (the hardwares given), we need to learn how to count how much of a memory a certain task will occupy. With the example code above, for a 4\*8 matrix, the memory occupied could be calculated as 4\*8\*4=128 bytes assuming we are using float32. 

This actually brings in a larger world: how numbers are stored in the computer. We are using IEEE 754 single precision standard, which is float32. It means we use 32 bits to store floating point numbers. The 0-22 elements store fraction. The 23-30 elements store exponent. The 31st element stores sign. The float64 is called 'double precision'. Here, we can understand that the 'single' from 'single precision' is a relative concept. 1 byte = 8 bits. Therefore, a floating point number under float32 occupies 4 bytes or 32 bits. That's where the '32' comes from. Let's try with codes:

```python
num_gpus = torch.cuda.device_count()
```

I am using a laptop equipped with one NVIDIA GeForce MX570 A. So the code returns 1. We could also use the following code to get detailed information for each GPU.

```python
for i in range(num_gpus):
    properties = torch.cuda.get_device_properties(i)
```

From the property we will have:
```
_CudaDeviceProperties(name='NVIDIA GeForce MX570 A', major=8, minor=6, total_memory=2047MB, multi_processor_count=16, uuid=cc46c636-447f-9cbd-5dc5-e4f6628de614, pci_bus_id=1, pci_device_id=0, pci_domain_id=0, L2_cache_size=1MB)
```

Let's try our best to understand everything. The name is obvious. 'major' and 'minor' are like generation numbers to define the computing capabilities. That is to say, we are now at the level of version 8.6 and we should avoid using CUDA functionalities higher than version 8.6. The definition must be made to the devide so CUDA would clearly know its limitations. 'multi_processor_count' is the number of multi-processors it has. Inside the currect devide, there are 16 multi-processors. The multi-processor, in the context of NVIDIA GPU, is also called Streaming Multiprocessor (SM). While my GeForce MX570 A has 16 multi-processors per full GPU with 128 CUDA Cores per SM (2,048 CUDA Cores in total), the powerful H100 has 144 SM per full GPU with 128 CUDA Cores per SM (18,432 CUDA Cores in total).

A very simple check could be done using:

```python
torch.cuda.is_available()
```

to see whether you already have an NVIDIA GPU to go.

Besides float64, float32 and float16, recent there have been bfloat 16, fp8 and its variant. What's the difference between float16 and bfloat16? The float16 is 'half precision'. The too low precision will cause instability in training. In the vanilla float16, 0-9 elements store fraction. 10-14 store exponent. The 15th element stores sign. The problem is then it provides limited dynamic ($2^{5}$ permutations). The bfloat16 adjusts it as: 0-6 (7 bits in total) as fraction, 7-14 as exponent and the 15th component as sign. The adjustment improves the dynamic. The shortened fraction might be an issue in science computing. But the precision is sufficient for deep learning as we could be a bit sloppy in deep learning. For storing optimizer states and parameters we still need float32.

### Lecture 3: Architectures, hyperparameters ###

## Assignments ##

### Assignment 1: Introduction to word vectors ###