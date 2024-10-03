# Llama-2 Implementation
### This is my Llama-2 Implementation for CMU's 10-423, Generative AI course.

The main portion of this code (and the differentiating factor of Llama-2 vs other LLMs) is the inclusion of Rotary Positional Embeddings (RoPE) and Grouped Query Attention (GQA).

RoPE aims to encode positional data into embeddings by rotating chunks of the embedding. Grouped Query Attention aims to minimize computational resource requirements by 'grouping' query heads with key & value heads in calculating attention.

The implementations for these can be found in `mingpt/model.py`.

To run this, simply run the `chargpt.py` file. The model will start training and outputting predictions every 100 iterations. Parameters can be set, like number of key/value heads, query heads, or the option to use RoPE.
