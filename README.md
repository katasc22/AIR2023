# AIR2023
## Experiment AIR23
### Comparing Different Approaches to Multlingual Retrieval
* Overcome language barriers
* Enhance user accessibility
### Research Questions
* What is the most convenient approach to retrieve documents from multilingual queries?
* How do different cross-lingual information retrieval approaches perform in comparison to each other?
* What are tradeoffs between the chosen methods?
## Getting started
pip install --upgrade ir_datasets

pip install -U sentence-transformers
## Dataset
[Vaswani](https://ir-datasets.com/vaswani.html)
* Small corpus of roughly 11,000 scientific abstracts
* English only (translation if necessary done by us)
## Conclusions
* MiniLM (monolingual approach) has the best overall performance.
* Translation-based approach also good results but there is longer execution time and more resources needed when comparing MBERT/distiluse with miniLM.
* MBERT does not produce good sentence embeddings out of the box.
* It is important to choose the correct model for your task, otherwise results and performance may vary.
## Contributors
### Katharina Aschauer
* Distiluse-base Approach
* Evaluation
* Plotting
### Maximilian Binder
* mBERT Approach
* Experiment Framework
* Preprocessing
### Jan-Peter Svetits
* MiniLM Approach
* Presentation
* Preprocessing
