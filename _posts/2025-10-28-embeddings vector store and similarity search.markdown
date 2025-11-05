---
layout: post
title:  "Containerization, similarity search and reading FAISS paper"
date:   2025-10-28 00:02:55 +0800
categories: AI
---

Embedding plays an important role in all of pre-training, Retrieval-Augmented Generation (RAG) and fine-tuning in the domain of AI. With the emergence of embeddings, needs to store and search embeddings soars. In terms of search, FAISS is a star showing up frequently in tech articles. Starting with a naive question "FAISS is mentioned everywhere but What on earth is FAISS?", I will be trying to figure it out a bit with the explorations recorded via this article.

Searching 'FAISS' in Google returns its [official documentation](https://ai.meta.com/tools/faiss/). Entering the documentation, I saw its full name as Facebook AI Similarity Search (FAISS). Therefore, we could infer that FAISS is mainly used for similarity search for after embedding extraction and storage. From the introduction, I know that FAISS uses Euclidean distance in the similarity check. In the FAISS paper, it's explicit that while also supporting some other distance metrices, Euclidean distance (L2) is by default used in FAISS.

However, the installation of FAISS poses me another question. It uses conda CLI to install and a GPU version requires CUDA 12.4, which is by the latest supported by Ubuntu 22.04 (not Ubuntu 24.04). I start to think that what if a developer needs multiple CUDA versions for different projects? Since CUDA is a toolkit instead of a Python module, we could not simply build different virtual environments to handle various project dependencies. Containerization is the key technique to overcome it. Today, when we talk about containerization, we will be rambling about either Docker or Kubernetes (K8s). In this article, I studies Docker containers as well as its VS Code extension - Dev Containers.

## Installation of FAISS and containerization

First of all, I would like to install FAISS. The official documentation recommends installation via conda. There are three versions available: `faiss-cpu`, `faiss-gpu` and `faiss-gpu-cuvs`. `faiss-cpu` uses only CPU. `faiss-gpu` enables both CPU and GPU for computation. `faiss-gpu-cuvs` is integrated with cuVS and collaborated on by NVIDIA and Meta. `faiss-gpu` and `faiss-gpu-cuvs` require specific CUDA Toolkit versions, 11.4/12.1 and 12.4 respectively. I use the most advanced `faiss-gpu-cuvs` with CUDA 12.4.

However, when I tried to install CUDA 12.4 in my WSL-Ubuntu 24.04 LTS, obviously Ubuntu 24.04 LTS does not support the installation of CUDA 12.4 since the version is already out of date. Additionally, I was very curious at a question that what if I have projects dependent on different CUDA Toolkit versions? After Googleing, I got two workarounds: (1) installing WSL-Ubuntu 22.04 LTS and install CUDA 12.4 in this new system; (2) installing Docker and configure Ubuntu 22.04 LTS inside the container without really installing a new system on my physical machine. I go ahead with (2). There are three reasons: (1) it's not a reasonable practice that developers switch systems just because of CUDA; (2) adding a new system is much more tedious than setting up containers; (3) I personally would like to learn containerization.

### Containers in Docker

As mentioned above, I started to think about such a question that what if my projects require different CUDA versions. For example, I have machine learning projects using CUDA 12.9, which is the latest CUDA version that my hardware could reach. But this FAISS requires CUDA 12.9. Could I install multiple versions on my host? And if I could, how could I switch between CUDA versions when I switch between corresponding development projects? Google search has given me a solution called containerization.

A container, as its name, contains everything necessary for a project, not at moduel level, language level or environment level, but at system level. A container is conceptually including but much larger than a python virtual environment or a language. Normally, a basic operating system (Linux), a programming language and some other development tools like CUDA and Git together configure a minimal production container. The functionality will be well-defined by an 'image' and several 'features'. They will be elaborated afterwards. A developer will feel easier if a CUDA 12.4 is installed in a container and a CUDA 12.9 is installed in another because containerization prevents contamination in the development environment. A container could be used for developing applications or separating files. A development container is a container used for application development.
 
 Since it's a lightweight system, there is a root user and some rootless users. A root user is the one with the highest permission and rootless users are assigned with different permission levels by the root. All the rootless users are administrated under `/home`. For instance, a new user created goes to `/home/devuser`. The developer user could create `/home/devuser/workspace` to build a workspace and start development. In a working directory `/home/devuser/workspace/project1/` the user `devuser` could place the virtual python environment (venv), .py files and requirements.txt inside this project folder. Below the conceptual framework of containerization is demonstrated.

![A conceptual framework of Docker](/assets/images/2025-10-28_docker_illustration.png)

#### Which Docker?

It's worth it to clarify roles played by Docker Engine, Docker CE/EE, Docker Build, Docker Desktop and Docker Compose. I was really confused by them and figured them out with some efforts.
- **Docker Engine**: it is the core empowering Docker applications. It is responsible for initializing, hosting and managing containers.
- **Docker CE/EE**: 'CE' is community edition. 'EE' is enterprise edition. Docker charges when a community of more than 200 members uses it. So approximatelty we could view their difference as the difference between a free version and a paid version.
- **Docker Desktop**: it's just a GUI wrapping Docker Engine up. On Windows we could download it. Since WSL does not have a GUI, it's not suggested even though a WSL-backed version is also supported.
- **Docker Compose**: Docker Engine creates containers. However, containers are created single. Docker Compose orchestrates containers for more complex applications. In a real production application, there are multiple containers communicating with each other. That is where Docker Compose jumps in.

For a simple purpose of tasting the flavor, it's enough to choose Docker Engine.

At the beginning, I followed [Docker's official instruction](https://docs.docker.com/desktop/setup/install/linux/ubuntu/#install-docker-desktop) to install Docker Engine in my WSL2-Ubuntu24.04 LTS. Since `faiss-gpu-cuvs` dependes on CUDA 12.4, I then need to initialize an empty Docker container so that I could install FAISS to that container later on.

Pragmatically, with Docker, a new container can be initialized (or created, equivalently) with the command below:
```bash
docker run [options]
```

As touched above, initialization of a container requires an image and its features. It's very difficult for me to understand the word 'image' semantically here. Actually, I am still at sea about the technologies behind the scene. As far as I understand now, broadly, an 'image' is some most important functionalities that a developer wants to include in that container. A container comes with nothing inside, even an operating system. Therefore, they should be defined in the 'image'.

#### Image

For this task to play with FAISS, since we want `faiss-gpu-cuvs`, a CUDA is necessary. We like Linux. We code in Python. Therefore, we could use an image of any one of the three primitives. It's more pratical to include two of them. For example, an image 'nvidia/cuda:12.4.0-devel-ubuntu22.04 bash' includes both CUDA Container Toolkit and Ubuntu22.04 like below. I did not name the container. The convention is, if a container name is not given, a random funny name will be given by Docker. The container is actually named as 'heuristic_banzai'.

```bash
docker run --gpus all -it nvidia/cuda:12.4.0-devel-ubuntu22.04 bash
```
The image is just the most basic configuration for a container. It's easily noted that Python is not configured yet. Therefore, we could add a specific Python version as a feature.

Of course, there are many ways to group the image and features. In the exemple command above, I used an image involving Ubuntu and CUDA, then I have to specify a Python version in 'features'. I could also specify a Python image and add CUDA to 'features'. As for Linux in this case, besides Pythonic images with the pure Python language, many images come with a basic Debian-based Linux and we need much patience to study the documentation!

#### CUDA Container Toolkit

We also need [CUDA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to configure a ceretain CUDA version required by that project. In this case, CUDA 12.4 is required by FAISS conda installation. One prerequisite to install CUDA Container Toolkit is NVIDIA GPU Driver. To test whether an NVIDIA GPU Driver already exists, we could run:

```bash
nvidia-smi
```

If a result like below is returned, we say a physical GPU is detected by the machine.

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.65                 Driver Version: 576.83         CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce MX570 A         On  |   00000000:01:00.0 Off |                  N/A |
| N/A   45C    P0              9W /   20W |       0MiB /   2048MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

Among the table given above, a trick here is 'CUDA Version: 12.9' on the upper right cornor. This is not the most up-to-date CUDA version installed. It only indicates which CUDA version it could be updated to with the current GPU infrastructure on the host.

#### Dev Containers

Dev Containers is the VS Code extension specifically for working with Docker Containers. I actually struggled pretty much with Dev Containers as I cannot properly connect to the VS Code Server. VS Code Server is the solution to working scnearios without GUI such as working with WSL and working with containers. 

Now the question comes - where to add the image and features? There are two ways:
- .devcontainer/devcontainer.json: I said the container is like a mini system and a developer user could create a workspace folder holding all the projects. Each project would go to a container. The structure of the devcontainer.json follows:
```
{
    "name": str,
    "image": str,
    "features": {
        {str: {option1: str, option2: str, ...}}
    },
    "customizations":,
    "remoteUser":,
}
```
- Dockerfile. This point will be studied in later articles.

One trick to remember is to add 'extensions' in the configuration file, especially 'Python' and 'Jupyter'. The final configuration looks like below:

```
{
	"workspaceFolder": "/home/devuser1/workspace/try_faiss",
	"remoteUser": "devuser1",
	"extensions": [
		"ms-python.debugpy",
		"ms-python.python",
		"ms-python.vscode-pylance",
		"ms-python.vscode-python-envs",
		"ms-toolsai.jupyter",
		"ms-toolsai.jupyter-keymap",
		"ms-toolsai.jupyter-renderers",
		"ms-toolsai.vscode-jupyter-cell-tags",
		"ms-toolsai.vscode-jupyter-slideshow"
	]
}
```

Then we need to run again command lines to install conda to the `devuser`. We will eventually download FAISS in a specific project folder under that rootless user instead of the root user.

## Similarity search

Finally we're talking about FAISS now. FAISS, as it claims itself, is not a vector database management system, not an embedding extractor. It's an open-source library implementing ANNS (Approximate Nearest Neighbor Search) algorithms. It only helps processing the intermediate representations and pushes the outcome into next steps (prediction, generation, etc.).

In general, ANNS is the opposite of brute-force search. Imaging we have 1M d-dimensional vectors in a vector database. Now here comes a query also embeded as a d-dimensional vector. To find which embedding is the most similar one with the query, a similarity search is conducted. A brute-force search will really calculate the distance between the query and every embedding from the database and pick the one with the shortest distance. ANNS will lower the accuracy a bit to find an approximate one or use many other ways.

The core of FAISS is the `Index` object. I was super confused about the naming. 'Index' means an indicator or reference. However, etymologically, the word means 'to point to/out'. That is to say, 'index' means to let things easier to be found. It makes sense now as FAISS, backed by ANNS, is much more efficient than looking for something without any clue like brute-force search. Let's follow the tutorial to taste the basics of FAISS implementation.

First of all, we create 100,000 64-dimentional database vectors and 10,000 dimentional query vectors using numpy's random number generator (`np.random.random` means numbers are drawn from a uniform distribution). The dimensionality is defined with the variable `d`. FAISS implicitly requires the same dimensionality across all database vectors and query vectors. There are techniques about the dimensionality unification, which will be addressed in later articles.

```python
import numpy as np

d = 64
nb = 100000
nq = 10000
np.random.seed(1234)
xb = np.random.random((nb,d)).astype('float32')
xb[:,0] += np.arange(nb)/1000.
xq = np.random.random((nq,d)).astype('float32')
xq[:,0] += np.arange(nq)/1000.
```
We follow a simple `IndexFlatL2` which basically uses brute-force search just to get familiar with basic FAISS operations.
- Step 1: Initialize an Index object with a compulsory argument dimensionality.
- Step 2: Add the vectors using the method add().
- Step 3: Search the `k Nearest Neighbors` with a specified `k`.
- Step 4: Two matrices will be returned. One integer matrix (denoted as `I`) is an integers matrix which indicates the indexes of found vectors. One floating number matrix (denoted as `D`) is an matrix which indicates the real distance between queries and database vectors.

An additional reminder is some functions in FAISS could store IDs of vectors. While IDs are not explicitly specified, ordinal input order will be the implicit IDs of vectors. However, the example `IndexFlatL2` here does not store IDs even implicitly.

```python
import faiss

k = 4
index = faiss.IndexFlatL2(d)
index.add(xb)
D, I = index.search(xq, k)
print(I)
print(D)
```

The outcome shows as following:

```
[[  381   207   210   477]
 [  526   911   142    72]
 [  838   527  1290   425]
 ...
 [11353 11103 10164  9787]
 [10571 10664 10632  9638]
 [ 9628  9554 10036  9582]]
[[6.815506  6.8894653 7.3956795 7.4290257]
 [6.6041145 6.679699  6.7209625 6.828678 ]
 [6.4703865 6.8578644 7.0043755 7.036564 ]
 ...
 [6.0727234 6.576782  6.6139526 6.7323   ]
 [6.6374817 6.6487427 6.8578796 7.009613 ]
 [6.2183533 6.4525146 6.548767  6.5812836]]
```

This is a first taste of FAISS altogether with some wandering through containerization.