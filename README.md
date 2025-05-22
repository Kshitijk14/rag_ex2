# rag_ex2

### Rag Ex. V1 Link: [rag_ex v1](https://github.com/Kshitijk14/rag_ex)

## Key tasks:
    1. dynamically update db
    2. evaluate workflow

## Quality of answers depends on:
    1. source material
    2. text splitting strategy
    3. LLM model f
       * or embedding (feature extraction)
       * & response generation (text generation) 
    4. prompt

## References:
1. Embedding Model:
   * [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
   * [langchain docs (for using BGE from Hugging Face)](https://python.langchain.com/docs/integrations/text_embedding/bge_huggingface/)
2. Generation Model:
   * [llama 3.2 3b](https://ollama.com/library/llama3.2)
   * [langchain docs](https://python.langchain.com/api_reference/ollama/llms/langchain_ollama.llms.OllamaLLM.html#ollamallm)
3. [Ref Video](https://www.youtube.com/watch?v=2TJxpyO3ei4), [Repo](https://github.com/pixegami/rag-tutorial-v2)

## Reference Papers:
* [Beyond English-Centric Multilingual Machine Translation](https://arxiv.org/abs/2010.11125)
* [Facebook FAIR's WMT19 News Translation Task Submission](https://arxiv.org/abs/1907.06616)
* [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934)
* [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
* [Embed Everything: A Method for Efficiently Co-Embedding Multi-Modal Spaces](https://arxiv.org/abs/2110.04599)