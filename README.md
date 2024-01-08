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
```
pip install -r requirements.txt```


```
air-experiment.py -h```

## Dataset
[Vaswani](https://ir-datasets.com/vaswani.html)
* Small corpus of roughly 11,000 scientific abstracts
* English only (translation if necessary done by us)
## Conclusions
* Monolingual model with pretranslated queries has good performance but is bad at scale.
* Choosing the right model for the respective task significantly impacts retrieval performance
* mBERT does not produce good sentence embeddings out of the box.
Best approach:
* Using fine-tuned multilingual model. It provides the best trade-offs between retrieval performance and computational requirements.
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
