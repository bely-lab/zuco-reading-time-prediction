# Predicting Sentence-Level Reading Time from Text

This project investigates whether sentence-level reading time can be predicted from textual features. The study uses eye-tracking data from the Zurich Cognitive Language Processing Corpus (ZuCo).

The goal is to compare different sentence representations and evaluate how well they predict mean sentence-level Total Reading Time (TRT).

## Dataset

The experiments use the **ZuCo corpus** (Hollenstein et al., 2018), specifically:

Task 1: Normal reading  
Modality: Eye-tracking

Sentence-level total reading time was computed by summing word-level TRT values and converting samples to milliseconds.

Reading times were averaged across 18 participants.

Final dataset size:
- 349 sentences

## Sentence Representations

Three sentence representations were compared:

**Structural baseline features**
- sentence length
- average word length
- lexical diversity
- long-word ratio

**TF–IDF representation**
- unigram and bigram lexical features

**Transformer embeddings**
- contextual sentence embeddings from BERT (bert-base-uncased)

All representations were evaluated using **Ridge regression**.

## Evaluation

Models were evaluated using **5-fold cross-validation**.

Metrics:
- Pearson correlation
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

## Main Results

| Model | Pearson r |
|------|------|
| Baseline features + Ridge | ~0.89 |
| Length-only model | ~0.88 |
| BERT embeddings | ~0.79 |
| TF-IDF | ~0.55 |

Sentence length accounts for most predictive power, while contextual embeddings did not outperform simpler structural features for sentence-level TRT prediction.

## Project Structure


## References

Hollenstein, N., et al. (2018).  
ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading.

Devlin, J., et al. (2019).  
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

Rayner, K. (1998).  
Eye movements in reading and information processing.
