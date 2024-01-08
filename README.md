# AIR2023
## Experiment AIR23
### Different Approaches to Multlingual Retrieval
Comparing the different approaches for cross-lingual information retrieval.
* How do different cross-lingual retrieval approaches perform in comparison to each other?
* Does a monolingual model that translates the query outperform a pretrained multlingual model?
* What are tradeoffs between the three different approaches we compared within our project?
## Getting started: 
pip install --upgrade ir_datasets
pip install -U sentence-transformers
## Dataset used
Vaswani
* Small corpus of roughly 11,000 scientific abstracts
* English only (translation if necessary done by us)
## Conclusions
* MiniLM (monolingual approach) has the best overall performance.
* Translation-based approach also good results but there is longer execution time and more resources needed when comparing MBERT/distiluse with miniLM.
* MBERT does not produce good sentence embeddings out of the box.
* It is important to choose the correct model for your task, otherwise results and performance may vary.
## Contributors
### Katharina Aschauer
* Knowledge Distillation
* Plotting
### Maximilian Binder
* MBERT
* Experiment Framework
### Jan-Peter Svetits
* Monolingual
* Presentation
