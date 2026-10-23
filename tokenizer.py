from collections import defaultdict
from functools import lru_cache
import re
from collections import Counter


def word_tokenize(text):
    return re.findall("[^\\W\\d_]+(?:['’ʼ][^\\W\\d_]+)*|\\d+|[^\\w\\s]|_", text.lower())


def test_word_tokenize():
    assert word_tokenize("Модель читає текст.") == ["модель", "читає", "текст", "."]
    assert word_tokenize("Комп'ютер і пам’ять") == ["комп'ютер", "і", "пам’ять"]
    assert word_tokenize("GPT-2: 50 токенів!") == [
        "gpt",
        "-",
        "2",
        ":",
        "50",
        "токенів",
        "!",
    ]
    assert word_tokenize("") == []


def build_word_vocab(texts):
    tokens = sorted({token for text in texts for token in word_tokenize(text)})
    return {"<pad>": 0, "<unk>": 1, **{token: i for i, token in enumerate(tokens, 2)}}


def encode_words(text, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in word_tokenize(text)]


def test_word_vocab():
    vocab = build_word_vocab(["Кіт спить.", "Пес спить."])
    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert encode_words("Кіт біжить.", vocab) == [vocab["кіт"], 1, vocab["."]]
    assert encode_words("Хом'як біжить.", vocab)[0:2] == [1, 1]


def char_ngrams(word, n=3):
    if type(n) is not int or n <= 0:
        raise ValueError("n має бути додатним цілим числом")
    if not word:
        return []
    bounded = ("^" + word + "$").ljust(n, "~")
    return [bounded[i : i + n] for i in range(len(bounded) - n + 1)]


def ngram_overlap(word_a, word_b, n=3):
    a, b = (set(char_ngrams(word_a, n)), set(char_ngrams(word_b, n)))
    return len(a & b) / len(a | b) if a | b else 1.0


def test_ngrams():
    assert char_ngrams("кіт", 3) == ["^кі", "кіт", "іт$"]
    assert char_ngrams("кіт", 2) == ["^к", "кі", "іт", "т$"]
    assert char_ngrams("", 3) == []
    assert char_ngrams("кіт", 6) == ["^кіт$~"]
    assert char_ngrams("кіт", 7) == ["^кіт$~~"]
    assert ngram_overlap("кіт", "пес", 6) == 0.0
    assert ngram_overlap("токенізатор", "токенізація", 3) > 0.4
    assert ngram_overlap("кіт", "трактор", 3) == 0.0
    try:
        char_ngrams("слово", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("n має бути додатним")


def build_ngram_vocab(texts, n=3):
    grams = set()
    for text in texts:
        for word in word_tokenize(text):
            grams.update(char_ngrams(word, n))
    vocab = {"<pad>": 0, "<unk>": 1}
    for gram in sorted(grams):
        vocab[gram] = len(vocab)
    return vocab


def encode_ngrams(text, vocab, n=3):
    return [
        vocab.get(gram, vocab["<unk>"])
        for word in word_tokenize(text)
        for gram in char_ngrams(word, n)
    ]


def test_ngram_vocab():
    vocab = build_ngram_vocab(["кіт"], 3)
    assert len(encode_ngrams("кіт", vocab)) == 3
    assert 1 not in encode_ngrams("кіт", vocab)
    assert 1 in encode_ngrams("cat", vocab)
    assert vocab["<pad>"] == 0 and vocab["<unk>"] == 1


def word_frequencies(texts):
    return Counter(
        (
            token
            for text in texts
            for token in word_tokenize(text)
            if token.replace("'", "").replace("’", "").replace("ʼ", "").isalpha()
        )
    )


def most_frequent_pair(vocabulary):
    counts = Counter()
    for symbols, frequency in vocabulary.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += frequency
    return min(counts, key=lambda pair: (-counts[pair], pair)) if counts else None


def merge_pair(symbols, pair):
    merged, i = ([], 0)
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


def _training_counts(vocabulary):
    words = list(vocabulary)
    counts = list(vocabulary.values())
    pair_counts = Counter()
    pair_words = defaultdict(set)
    symbol_counts = Counter()
    for index, symbols in enumerate(words):
        count = counts[index]
        for symbol in symbols:
            symbol_counts[symbol] += count
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += count
            pair_words[pair].add(index)
    return (words, counts, pair_counts, pair_words, symbol_counts)


def _merge_training_pair(state, pair, merge):
    words, counts, pair_counts, pair_words, symbol_counts = state
    for index in list(pair_words[pair]):
        old = words[index]
        new = merge(old, pair)
        count = counts[index]
        for symbol, amount in Counter(old).items():
            symbol_counts[symbol] -= amount * count
        for symbol, amount in Counter(new).items():
            symbol_counts[symbol] += amount * count
        for old_pair, amount in Counter(zip(old, old[1:])).items():
            pair_counts[old_pair] -= amount * count
            pair_words[old_pair].discard(index)
            if pair_counts[old_pair] == 0:
                del pair_counts[old_pair]
        for new_pair, amount in Counter(zip(new, new[1:])).items():
            pair_counts[new_pair] += amount * count
            pair_words[new_pair].add(index)
        words[index] = new


def train_bpe(texts, num_merges):
    if type(num_merges) is not int or num_merges < 0:
        raise ValueError("кількість злиттів має бути невід’ємним цілим числом")
    vocabulary = {}
    for word, count in word_frequencies(texts).items():
        vocabulary[tuple(word) + ("</w>",)] = count
    state = _training_counts(vocabulary)
    pair_counts = state[2]
    merges = []
    for _ in range(num_merges):
        if not pair_counts:
            break
        pair = min(pair_counts, key=lambda p: (-pair_counts[p], p))
        _merge_training_pair(state, pair, merge_pair)
        merges.append(pair)
    return merges


def apply_bpe(word, merges):
    if not word:
        return []
    symbols = tuple(word) + ("</w>",)
    for pair in merges:
        symbols = merge_pair(symbols, pair)
    return list(symbols)


def test_bpe():
    vocabulary = {("м", "а", "м", "а", "</w>"): 2, ("м", "а", "л", "а", "</w>"): 1}
    assert most_frequent_pair(vocabulary) == ("м", "а")
    assert merge_pair(("м", "а", "м", "а", "</w>"), ("м", "а")) == ("ма", "ма", "</w>")
    merges = train_bpe(["мама мама мала тон"], 3)
    assert merges[0] == ("м", "а")
    assert apply_bpe("мама", merges) == ["мама</w>"]
    unseen = apply_bpe("мамонт", merges)
    assert "<unk>" not in unseen
    assert "".join(unseen).replace("</w>", "") == "мамонт"


def average_bpe_length(texts, merges):
    frequencies = word_frequencies(texts)
    total_words = sum(frequencies.values())
    if not total_words:
        raise ValueError("Корпус без слів")
    return (
        sum(
            (
                count * len(apply_bpe(word, merges))
                for word, count in frequencies.items()
            )
        )
        / total_words
    )


def test_average_bpe_length():
    corpus = ["токен токени токенізація", "токенізатор токенізує текст"]
    merges = train_bpe(corpus, 20)
    before = average_bpe_length(corpus, [])
    after = average_bpe_length(corpus, merges)
    assert after < before
    assert before > 1


def wordpiece_symbols(word):
    return tuple(
        (character if i == 0 else "##" + character for i, character in enumerate(word))
    )


def wordpiece_pair_scores(vocabulary):
    token_counts, pair_counts = (Counter(), Counter())
    for symbols, count in vocabulary.items():
        for symbol in symbols:
            token_counts[symbol] += count
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += count
    return {
        pair: count / (token_counts[pair[0]] * token_counts[pair[1]])
        for pair, count in pair_counts.items()
    }


def merge_wordpiece_pair(symbols, pair):
    merged, i = ([], 0)
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            right = (
                symbols[i + 1][2:]
                if symbols[i + 1].startswith("##")
                else symbols[i + 1]
            )
            merged.append(symbols[i] + right)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


def train_wordpiece(texts, vocab_size):
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("розмір словника має бути додатним цілим числом")
    vocabulary = {
        wordpiece_symbols(word): count
        for word, count in word_frequencies(texts).items()
    }
    vocab = {"[UNK]"}
    for symbols in vocabulary:
        vocab.update(symbols)
    if len(vocab) > vocab_size:
        raise ValueError("словник менший за початковий алфавіт")
    state = _training_counts(vocabulary)
    pair_counts = state[2]
    symbol_counts = state[4]
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        pair = min(
            pair_counts,
            key=lambda p: (
                -pair_counts[p] / (symbol_counts[p[0]] * symbol_counts[p[1]]),
                p,
            ),
        )
        merged_token = pair[0] + pair[1].removeprefix("##")
        _merge_training_pair(state, pair, merge_wordpiece_pair)
        vocab.add(merged_token)
    return vocab


def apply_wordpiece(word, vocab, unk_token="[UNK]"):
    pieces, start = ([], 0)
    while start < len(word):
        end = len(word)
        while end > start:
            candidate = ("##" if start else "") + word[start:end]
            if candidate in vocab:
                break
            end -= 1
        if end == start:
            return [unk_token]
        pieces.append(candidate)
        start = end
    return pieces


def test_wordpiece():
    weighted_vocab = {("а", "##б"): 2, ("а", "##в"): 1, ("г", "##в"): 2}
    scores = wordpiece_pair_scores(weighted_vocab)
    assert scores["а", "##б"] > scores["а", "##в"]
    assert merge_wordpiece_pair(("м", "##а", "##м", "##а"), ("м", "##а")) == (
        "ма",
        "##м",
        "##а",
    )
    vocab = train_wordpiece(["мама мама мала тон"], 30)
    assert "[UNK]" in vocab
    assert apply_wordpiece("мама", vocab) == ["мама"]
    bert_vocab = {"[UNK]", "ток", "токен", "##ен", "##ізація"}
    assert apply_wordpiece("токенізація", bert_vocab) == ["токен", "##ізація"]
    assert apply_wordpiece("токенx", bert_vocab) == ["[UNK]"]
    assert apply_wordpiece("", bert_vocab) == []
    try:
        train_wordpiece(["текст"], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("vocab_size має бути додатним")


from datasets import load_dataset

FLORES_LANGUAGES = {
    "English": "eng_Latn",
    "Українська": "ukr_Cyrl",
    "ქართული": "kat_Geor",
}


def load_flores_dev():
    corpora = {}
    for language, config in FLORES_LANGUAGES.items():
        dataset = load_dataset(
            "tomasmajercik/flores-parquet",
            config,
            split="validation",
            revision="dde46bb7d02cc6a45e9568d2ddb4fa6585412434",
        )
        corpora[language] = list(dataset["sentence"])
    return corpora


VOCAB_SIZES = (500, 1000, 2000)


def bpe_num_merges(texts, vocab_size):
    frequencies = word_frequencies(texts)
    characters = {character for word in frequencies for character in word}
    initial_vocab_size = len(characters) + 1
    if type(vocab_size) is not int or vocab_size < initial_vocab_size:
        raise ValueError("vocab_size менший за початковий алфавіт")
    return vocab_size - initial_vocab_size


def apply_bpe_to_text(text, merges):
    return [
        piece for token in word_tokenize(text) for piece in apply_bpe(token, merges)
    ]


def apply_wordpiece_to_text(text, vocab):
    return [
        piece
        for token in word_tokenize(text)
        for piece in apply_wordpiece(token, vocab)
    ]


import tiktoken


def fertility(tokenize, texts):
    total_tokens = 0
    total_words = 0
    for text in texts:
        total_tokens += len(tokenize(text))
        total_words += len(text.split())
    if total_words == 0:
        raise ValueError("у корпусі немає слів")
    return total_tokens / total_words


def test_fertility():
    assert fertility(list, ["aa b", "ccc"]) == 7 / 3
    try:
        fertility(list, ["", "   "])
    except ValueError:
        pass
    else:
        raise AssertionError("Корпус без слів має давати ValueError")


def cached_subword_tokenizer(tokenize_word):

    @lru_cache(maxsize=50000)
    def tokenize_cached(word):
        return tuple(tokenize_word(word))

    def tokenize_text(text):
        return [
            piece for word in word_tokenize(text) for piece in tokenize_cached(word)
        ]

    return tokenize_text


def fertility_table(tokenizers, corpora):
    return {
        name: {
            language: fertility(tokenize, texts) for language, texts in corpora.items()
        }
        for name, tokenize in tokenizers.items()
    }


import matplotlib.pyplot as plt
import numpy as np

MIXED_CASES = [
    "Я навчаю tokenizer на FLORES dataset.",
    "The model розуміє ქართულ ენას?",
    "Київ -> Tbilisi -> თბილისი -> London 🌍",
    "Замовлення №2026-09-24 коштує $19.99.",
    "support@example.com або https://example.com/допомога",
    "я дуже втомився від курсу основи Large Language Models",
]


def compare_mixed_cases(texts, tokenizers):
    for index, text in enumerate(texts, 1):
        print(f"{index}. {text}")
    print()
    print(f"{'tokenizer':<18}" + "".join((f"{i:>7}" for i in range(1, len(texts) + 1))))
    for name, tokenize in tokenizers.items():
        counts = [len(tokenize(text)) for text in texts]
        print(f"{name:<18}" + "".join((f"{count:>7}" for count in counts)))


if __name__ == "__main__":
    test_word_tokenize()
    print("✓ word_tokenize")
    print(word_tokenize("Комп’ютер навчається!"))
    test_word_vocab()
    print("✓ word vocabulary")
    demo_word_vocab = build_word_vocab(["Кіт спить.", "Пес спить."])
    print("Кіт біжить. →", encode_words("Кіт біжить.", demo_word_vocab))
    test_ngrams()
    print("✓ character n-grams")
    print("Триграми:", char_ngrams("кіт", 3))
    print("Шестиграми з padding:", char_ngrams("кіт", 6))
    test_ngram_vocab()
    print("✓ n-gram vocabulary")
    test_bpe()
    print("✓ BPE")
    demo_merges = train_bpe(["мама мама мала тон"], 3)
    print("BPE, мамонт:", apply_bpe("мамонт", demo_merges))
    test_average_bpe_length()
    print("✓ BPE reduces average sequence length")
    test_wordpiece()
    print("✓ WordPiece")
    demo_wordpiece = train_wordpiece(["мама мама мала тон"], 30)
    print("WordPiece, мамонт:", apply_wordpiece("мамонт", demo_wordpiece))
    flores = load_flores_dev()
    assert all((len(texts) == 997 for texts in flores.values()))
    train_texts = [
        text for language_texts in flores.values() for text in language_texts
    ]
    print({language: len(texts) for language, texts in flores.items()})
    print("Усього навчальних рядків:", len(train_texts))
    example_index = 0
    for language, texts in flores.items():
        print(f"{language}: {texts[example_index]}\n")
    bpe_models = {}
    wordpiece_models = {}
    for vocab_size in VOCAB_SIZES:
        bpe_models[vocab_size] = train_bpe(
            train_texts, bpe_num_merges(train_texts, vocab_size)
        )
        wordpiece_models[vocab_size] = train_wordpiece(train_texts, vocab_size)
    print("Навчено BPE і WordPiece для:", VOCAB_SIZES)
    word_vocab = build_word_vocab(train_texts)
    trigram_vocab = build_ngram_vocab(train_texts, 3)
    gpt_encodings = {
        "GPT-2": tiktoken.encoding_for_model("gpt-2"),
        "GPT-3.5": tiktoken.encoding_for_model("gpt-3.5-turbo"),
        "GPT-4": tiktoken.encoding_for_model("gpt-4"),
        "GPT-5": tiktoken.encoding_for_model("gpt-5"),
    }
    for name, encoding in gpt_encodings.items():
        print(f"{name:7} -> {encoding.name:12} ({encoding.n_vocab} tokens)")
    test_fertility()
    print("✓ fertility")
    tokenizers_to_compare = {}
    for size, merges in bpe_models.items():
        tokenizers_to_compare[f"BPE-{size}"] = cached_subword_tokenizer(
            lambda word, merges=merges: apply_bpe(word, merges)
        )
    for size, vocab in wordpiece_models.items():
        tokenizers_to_compare[f"WordPiece-{size}"] = cached_subword_tokenizer(
            lambda word, vocab=vocab: apply_wordpiece(word, vocab)
        )
    for name, encoding in gpt_encodings.items():
        tokenizers_to_compare[name] = encoding.encode
    results = fertility_table(tokenizers_to_compare, flores)
    header = f"{'tokenizer':<24}" + "".join((f"{lang:>14}" for lang in flores))
    print(header)
    for name, values in results.items():
        print(f"{name:<24}" + "".join((f"{values[lang]:14.3f}" for lang in flores)))
    baseline_tokenizers = {
        "Words": lambda text: encode_words(text, word_vocab),
        "Trigrams": lambda text: encode_ngrams(text, trigram_vocab, 3),
    }
    baseline_results = fertility_table(baseline_tokenizers, flores)
    print("Словник слів:", len(word_vocab))
    print("Словник триграм:", len(trigram_vocab))
    for name, values in baseline_results.items():
        print(name, {lang: round(value, 3) for lang, value in values.items()})
    alphabet = set("".join(word_frequencies(train_texts))) | {"</w>"}
    for size in VOCAB_SIZES:
        actual_bpe = len(alphabet | {a + b for a, b in bpe_models[size]})
        print(f"Ціль {size}: BPE {actual_bpe}, WordPiece {len(wordpiece_models[size])}")
    unknown_rates = {}
    for language, texts in flores.items():
        units = [token for text in texts for token in word_tokenize(text)]
        lexical = [
            token
            for token in units
            if token.replace("'", "").replace("’", "").replace("ʼ", "").isalpha()
        ]
        unknown_rates[language] = {}
        for size, vocab in wordpiece_models.items():
            unknown = sum(
                (apply_wordpiece(token, vocab) == ["[UNK]"] for token in units)
            )
            lexical_unknown = sum(
                (apply_wordpiece(token, vocab) == ["[UNK]"] for token in lexical)
            )
            all_rate = 100 * unknown / len(units) if units else 0.0
            word_rate = 100 * lexical_unknown / len(lexical) if lexical else 0.0
            unknown_rates[language][size] = (all_rate, word_rate)
            print(
                f"{language}, WP-{size}: UNK усіх {all_rate:.2f}%, слів {word_rate:.2f}%"
            )
    names = list(results)
    languages = list(flores)
    values = np.array([[results[name][lang] for lang in languages] for name in names])
    fig, ax = plt.subplots(figsize=(7, 5.4), layout="constrained")
    image = ax.imshow(values, cmap="YlOrRd", vmin=0, aspect="auto")
    ax.set_xticks(range(len(languages)), ["English", "Ukrainian", "Georgian"])
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Fertility на FLORES dev: 997 речень на мову")
    for i in range(len(names)):
        for j in range(len(languages)):
            color = "white" if values[i, j] > values.max() * 0.62 else "black"
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, label="токенів на слово", shrink=0.85)
    plt.show()
    compare_mixed_cases(MIXED_CASES, tokenizers_to_compare)
    for text in ["токенізація", "Київ თბილისი 🌍"]:
        print("\nТекст:", text)
        print("BPE-2000:", apply_bpe_to_text(text, bpe_models[2000]))
        print("WordPiece-2000:", apply_wordpiece_to_text(text, wordpiece_models[2000]))
