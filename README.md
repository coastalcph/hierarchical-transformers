# Hi-Transformers (Hierarchical Transformers)

A simplified and modular re-implementation of the architecture proposed in the work "Hi-Transformer: Hierarchical Interactive Transformer for Efficient and Effective Long Document Modeling" by Wu et al. (2021) (https://aclanthology.org/2021.acl-short.107/).

<img src="hi-transformers.png"/>

The repository supports several variants of the `Hi-Transformer` architecture. The specific layout of stacking (interleaving) sentence (S) and document (D) encoders can be specified by the `encoder_layout` parameter. For example:

* **_Symmetric Interleaved_** (1-by-1) sentence (S), document (D) encoders, e.g., a 6-layer model has 12 effective transformer blocks (Layout: SD/SD/SD/SD/SD/SD).

```json
"encoder_layout": {
"0": {"sentence_encoder": true, "document_encoder":  true},
"1": {"sentence_encoder": true, "document_encoder":  true},
"2": {"sentence_encoder": true, "document_encoder":  true},
"3": {"sentence_encoder": true, "document_encoder":  true},
"4": {"sentence_encoder": true, "document_encoder":  true},
"5": {"sentence_encoder": true, "document_encoder":  true}},
```

* **_Non-symmetric Interleaved_** sentence (S), document (D) encoders, where pairing (or skipping) document encoders block by block, e.g., a 6-layer model and 8 effective transformer blocks (Layout: S/S/SD/S/S/SD).

```json
"encoder_layout": {
"0": {"sentence_encoder": true, "document_encoder":  false},
"1": {"sentence_encoder": true, "document_encoder":  false},
"2": {"sentence_encoder": true, "document_encoder":  true},
"3": {"sentence_encoder": true, "document_encoder":  false},
"4": {"sentence_encoder": true, "document_encoder":  false},
"5": {"sentence_encoder": true, "document_encoder":  true}},
```

* **_Non-symmetric (Early/Late) Interleaved_** sentence (S), document (D) encoders, where pairing (or skipping) document encoders block by block differs between early or late stages (blocks), e.g., a 6-layer model and 8 effective transformer blocks (Layout: S/S/S/S/SD/SD).


```json
"encoder_layout": {
"0": {"sentence_encoder": true, "document_encoder":  false},
"1": {"sentence_encoder": true, "document_encoder":  false},
"2": {"sentence_encoder": true, "document_encoder":  false},
"3": {"sentence_encoder": true, "document_encoder":  false},
"4": {"sentence_encoder": true, "document_encoder":  true},
"5": {"sentence_encoder": true, "document_encoder":  true}},
```

### Requirements

Make sure that all required packages are installed:

```
torch>=1.11.0
transformers>=4.15.0
datasets>=2.0.0
tokenizers>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.62.0
```

### How to run experiments?

So far, we are testing the core implementations, you can play around with the following script:

* `Hi-Transformer`: `/models/hi-transformer/validate_hi_transformer.py`

You can also try to train a new LM based on `Hi-Transformer`:

* `MLM`: `train_hi_transformer_mlm.sh`
* `Multi-Objective Pretraining (MLM, DRP, SRP)`: `train_hi_transformer_.sh`


Try on Google Colab: https://colab.research.google.com/drive/15feh49wqBshgkcvbO6QypvJoa3dG6P5S?usp=sharing

### I still have open questions...

Please post your question on [Discussions](https://github.com/coastalcph/hi-transformers/discussions) section or communicate with the corresponding author via e-mail.
