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

# Evaluation Pipeline:

## Retrieval Metrics
   * **Mean Reciprocal Rank (MRR)**
   * **Normalized Discounted Cumulative Gain (nDCG@k)**
   * **Precision@k**
   * **Recall@k**

## Generation Metrics
   ### Initializing with only F1 score for logical understanding
   * **f1 score** → 
     * Overall token-level similarity (0.0-1.0)
     * measures how well the generated answer matches the ground truth answer at the token level
     * harmonic mean of precision and recall
     * 2 * (Precision * Recall) / (Precision + Recall)
   * **precision** → what fraction of generated tokens are relevant?
   * **recall** → what fraction of relevant tokens were generated?

   ### remaining generation evaluation metrics
   * **em (exact match)** → 
     * Binary perfect match (0 or 1)
     * binary score - 1 if predicted answer exactly matches ground truth (after normalization), 0 otherwise
   * **ROUGE-L** → 
     * Sequence-based similarity (0.0-1.0)
     * measures longest common subsequence between predicted and ground truth
   * **BLEU** → 
     * N-gram overlap with brevity penalty (0.0-1.0)
     * somewhat similar to ROUGE-N
     * measures n-gram overlap, commonly used in translation/generation tasks
   * **faithfulness (%)** → 
     * How much is supported by context (0.0-1.0, 100% = fully supported)
     * measures how much of the generated answer can be supported by the retrieved context


# References:
1. Embedding Model:
   * [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
   * [langchain docs (for using BGE from Hugging Face)](https://python.langchain.com/docs/integrations/text_embedding/bge_huggingface/)
2. Generation Model:
   * [llama 3.2 3b](https://ollama.com/library/llama3.2)
   * [langchain docs](https://python.langchain.com/api_reference/ollama/llms/langchain_ollama.llms.OllamaLLM.html#ollamallm)
3. [Ref Video](https://www.youtube.com/watch?v=2TJxpyO3ei4), [Repo](https://github.com/pixegami/rag-tutorial-v2)
4. Generation Metrics:
   * [EM](https://huggingface.co/spaces/evaluate-metric/exact_match)
   * 

## Articles:
* [Understanding LLM Evaluation and Benchmarks: A Complete Guide](https://www.turing.com/resources/understanding-llm-evaluation-and-benchmarks)
* [What are LLM benchmarks?](https://www.ibm.com/think/topics/llm-benchmarks#:~:text=Exact%20match%20is%20the%20proportion,is%20at%20comprehending%20a%20task.)
* [Evaluating Large Language Models – Evaluation Metrics](https://www.enkefalos.com/newsletters-and-articles/evaluating-large-language-models-evaluation-metrics/#:~:text=Automatic%20Evaluation%20Metrics:,jumps%20over%20the%20lazy%20dog%22.)

## Papers:
* [Beyond English-Centric Multilingual Machine Translation](https://arxiv.org/abs/2010.11125)
* [Facebook FAIR's WMT19 News Translation Task Submission](https://arxiv.org/abs/1907.06616)
* [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934)
* [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
* [Embed Everything: A Method for Efficiently Co-Embedding Multi-Modal Spaces](https://arxiv.org/abs/2110.04599)
* [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://nlp.stanford.edu/pubs/rajpurkar2016squad.pdf)