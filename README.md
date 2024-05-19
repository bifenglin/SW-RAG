# SW-RAG

SW-RAG is an innovative system designed to improve the factual accuracy and reliability of outputs from Large Language Models (LLMs) by incorporating a sliding window mechanism during the indexing phase of the retrieval process. This repository contains the implementation of SW-RAG, along with instructions for usage and additional resources.

## Note
- [rag-chroma-private](packages%2Frag-chroma-private): Implemented single query RAG with three sliding window strategies, and the sliding window split is located in the splitter folder.
- [rag-ollama-multi-query](packages%2Frag-ollama-multi-query): Implemented multi query RAG with three sliding window strategies, and the sliding window split is located in the splitter folder.

## Features
- Implementation of the SW-RAG system architecture
- Support for multiple sliding window segmentation strategies (Fixed Window Size and Fixed Step Length Split, Dynamic Window Size with Fixed Step Length Split, Dynamic Window Size and Dynamic Step Length Split)
- Integration with popular vector databases for efficient document storage and retrieval
- Evaluation scripts for assessing system performance on various datasets and query types
- Parameter optimization tools for selecting optimal chunk size and step size settings

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/your_username/sw-rag.git
```

Navigate to the cloned directory:
```bash
cd sw-rag
```

Install the LangChain CLI if you haven't yet

```bash
pip install -U langchain-cli
```

## Setup LangSmith (Optional)
LangSmith will help us trace, monitor and debug LangChain applications. 
LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/). 
If you don't have access, you can skip this section


```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

## Launch LangServe

```bash
langchain serve
```

## Running in Docker

This project folder includes a Dockerfile that allows you to easily build and host your LangServe app.

### Building the Image

To build the image, you simply:

```shell
docker build . -t my-langserve-app
```

If you tag your image with something other than `my-langserve-app`,
note it for use in the next step.

### Running the Image Locally

To run the image, you'll need to include any environment variables
necessary for your application.

In the below example, we inject the `OPENAI_API_KEY` environment
variable with the value set in my local environment
(`$OPENAI_API_KEY`)

We also expose port 8080 with the `-p 8080:8080` option.

```shell
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8080:8080 my-langserve-app
```
