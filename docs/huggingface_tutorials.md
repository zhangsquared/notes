# HuggingFace Tutorials

## [Transformers](https://youtu.be/H39Z_720T5s)

Transforers consists of `Encoder` and `Decoder`.

The `encoder` encodes test into numerical representations

    * the numerical representations are also called as `embeddings` or `features`
    * `bi-directional` properties
    * use `self-attention` mechanism

The `decoder` decodes the represetnations from the encoder. I can also accepts text inputs

    * use `masked self-attention` mechanism
    * `uni-directional` property
    * used in an `auto-regressive` manner

Encoder-decoder transformers == Sequence-to-sequence transformers

## [Tokenizers](https://youtu.be/VFp38yj8h3A?si=GD7nYxwRkGjZyb4I)

Tokenize => Model => PostProcessing

RawText => InputIDs => logits => Predictions

The tokenzier's objective is to find a meaningful representation

1. Word based
    * very large vocabularies
    * large quntity of out-of-vacabulary tokens
    * loss of meaning across very similar words
2. Char based
    * very long sequences
    * less meaningful individual tokens
3. Subword based (most popular)
    * Frequently used words should not be split into smaller subwords
    * Rare words should be decomposed into meaningful subwords
    e.g. `dog` =X=> `d`, `o`, `g`, `dogs` => `dog`, `s`

## Datasets

Datasets is a library for easily accessing and sharing datasets for Audio, Computer Vision, and Natural Language Processing (NLP) tasks.

[Load a dataset from the Hub](https://huggingface.co/docs/datasets/load_hub)

[Proprocess](https://huggingface.co/docs/datasets/use_dataset)

Sometimes you may need to rename a column, and other times you might need to unflatten nested fields.

## [NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1)



