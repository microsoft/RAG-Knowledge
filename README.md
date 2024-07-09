# README

This project provides tips and techniques for RAG (Retrieval-Augmented Generation).

## Purpose

- This repository provides practical design and knowledge for RAG (Retrieval-Augmented Generation).
- Specifically, it breaks down RAG into components and provides core design elements, techniques, and practical guidance for each.
- Additionally, by offering implementation sample code on Azure, it aims to promote a deeper understanding and accelerate the development speed for users.

### Focus Components

This project focuses on several key areas:

1. **Data Preprocessing for RAG**: This includes extracting text data from various document formats (PDF, Word, HTML), Text Standardization and Normalization, and Chunking Optimization for RAG.
2. **Chunking Optimization**: In RAG, consider dividing the original document into appropriate sizes (chunking) when inputting data into the search index, taking into account search efficiency and the processing performance of large language models. While recent large language models have significantly increased token input capacity, it is recommended to consider RAG and accurate chunking since accuracy may decrease as context length increases.
3. **Index Design for Retriever**: This involves creating efficient and scalable indices to improve retrieval speed and accuracy, choosing the right indexing structures, and handling updates to the index.
4. **Query Optimization**: This includes techniques for crafting effective queries using various search methods such as keyword-based, vector-based, and hybrid approaches. It also covers query expansion techniques to enhance search results such as Hypothetical Document Embeddings (HyDE) methods.
5. **Evaluation**: This involves setting up benchmarks, selecting appropriate evaluation metrics, and conducting experiments to measure the performance of the RAG system.

## Background

- RAG has been extensively developed and validated as a core architecture for generative AI applications.
- While the concept of RAG is quite simple, achieving applications with usable accuracy levels involves many considerations, and many projects face challenges related to accuracy and development productivity.
- Numerous techniques and tools have been recently introduced to enhance the accuracy and performance of RAG. However, a systematic compilation of when to use these techniques and how to implement them is not readily available.

## When Should You Use This Repo?

- When you want to understand the basic concepts and architecture of RAG in the early stages of consideration.
- When you need to learn the detailed design elements and techniques for each component of RAG.
- When facing accuracy and performance challenges in the development and evaluation of RAG.
- When you want to refer to implementation samples for various RAG techniques.

## Roadmap of Provided Content

- Implementation sample code for various components and techniques.
- End-to-End RAG applications.
  - Examples of manual documentation (text only): Developer documentation for Azure AI Search.
  - Examples of manual documentation (text and images): Developer documentation for Azure AI Search.
  - Examples of research papers: A collection of papers on LLM and RAG techniques.
- Design guidance for performance improvement using the latest features of Azure AI Search.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.