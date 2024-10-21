# Speech MI - DJ 2024 Fall Project

## TODOs
1. topdown design of experiment script
    - two tasks: classification / forecasting
    - frameworks:
        - text only (interleaved text)
        - text + last utt text & speech (interleaved)
        
    
    
## Model and Framework
- LLM + Speech Adapter
- We can use the srag codebase...
    - 7b
    - Hubert Encoder Adapter
- Learning Objective (https://arxiv.org/pdf/2406.05968)
    - Next Token Prediction (NTP)
    - token logit distillation (LD)
- How to train them
    - Unified model with input formatting
    - Separet model for each tasks

### Experiment 1: Counselor Response Prediction

### Experiment 2: Client Modeling



## RQ & Hypothesis **
- Which framework: Speech-based MI and Conversation Modeling that consumses client utterance in **speech form, not transcribed form**,
- is different how: it's better for MI perfornace on tasks:
    - counselor response prediction
    - client understanding
    - etc
- from which previous baseline: text only
- because of which aspect
    - paralinguistic information and stuff
    - but how show


## Which Specific Tasks?

Two tasks I came up with
- Counselor Response Prediction (perplexity? text similarity?)
- Counseling Quality Estimation
- Prediction of Client Attitude and stuff
    - Client talk type


## Man Repo
https://www.overleaf.com/project/66d8ad52258808d4782fe8df