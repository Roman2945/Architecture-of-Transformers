# %% [markdown]
# # Токенізація
#
# Мовна модель працює не з текстом, а з послідовністю чисел. Тому перед входом
# у трансформер текст проходить важливе перетворення:
#
# 1. **токенізатор** ділить текст на токени та замінює їх ідентифікаторами;
#
# У цій лабораторній ми пройдемо шлях від токенів-слів до BPE і WordPiece,
# а потім побачимо, чому самих токенів недостатньо для трансформера. Усі алгоритми
# реалізуємо з нуля, використовуючи лише стандартну бібліотеку Python.

# %% [markdown]
# ## 1. Токенізатор на рівні слів
#
# Найпростіша ідея — вважати кожне слово окремим токеном. Спочатку приведемо
# текст до нижнього регістру, відділимо розділові знаки й збережемо український
# апостроф усередині слова.
#
# ```python
# word_tokenize("Комп'ютер навчається!")
# # ["комп'ютер", "навчається", "!"]
# ```
#
# Така нормалізація зменшує словник: `Модель` і `модель` стають одним токеном.
# Водночас ми втрачаємо частину інформації. У реальних токенізаторах правила
# нормалізації є окремим важливим рішенням.

# %%

def word_tokenize(text):
    """ділить текст на слова, числа і знаки"""
    import re
    return re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+|[^\w\s]|_", text.lower())


def test_word_tokenize():
    assert word_tokenize("Модель читає текст.") == [
        "модель", "читає", "текст", "."
    ]
    assert word_tokenize("Комп'ютер і пам’ять") == [
        "комп'ютер", "і", "пам’ять"
    ]
    assert word_tokenize("GPT-2: 50 токенів!") == [
        "gpt", "-", "2", ":", "50", "токенів", "!"
    ]
    assert word_tokenize("") == []


if __name__ == "__main__":
    test_word_tokenize()
    print("✓ word_tokenize")

# %% [markdown]
# ### Словник та проблема OOV
#
# Побудуємо словник тільки за навчальним корпусом. Два спеціальні токени:
#
# - `<pad>` доповнює послідовності до однакової довжини;
# - `<unk>` замінює невідоме слово (**out of vocabulary**, OOV).
#
# Якщо модель бачила `кіт`, але не бачила `котик`, усе слово `котик` стане
# `<unk>`. Різні невідомі слова отримають однаковий id — їх уже неможливо
# відрізнити.

# %%

def build_word_vocab(texts) -> dict[str, int]:
    """збирає словник зі слів навчального корпусу"""
    words = set()
    for text in texts:
        words.update(word_tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word in sorted(words):
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode_words(text, vocab):
    """замінює токени на їхні id"""
    return [vocab.get(word, vocab["<unk>"]) for word in word_tokenize(text)]


def test_word_vocab():
    vocab = build_word_vocab(["Кіт спить.", "Пес спить."])
    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert encode_words("Кіт біжить.", vocab) == [vocab["кіт"], 1, vocab["."]]
    assert encode_words("Хом'як біжить.", vocab)[0:2] == [1, 1]


if __name__ == "__main__":
    test_word_vocab()
    print("✓ word vocabulary")

# Питання:
# - Що станеться зі словником, якщо не привести слова до нижнього регістру?
# - Чому великий словник збільшує кількість параметрів embedding-таблиці?
# - Яку інформацію втрачено у двох послідовностях `[1, 1, id_крапки]`?

# %% [markdown]
# ## 2. Символьні n-грами
#
# Замість цілого слова можна взяти всі послідовності з `n` сусідніх символів.
# Додамо маркери `^` і `$`, щоб відрізняти початок та кінець слова:
#
# ```python
# char_ngrams("кіт", 3) == ["^кі", "кіт", "іт$"]
# ```
#
# У споріднених слів є спільні частини. Наприклад, навіть якщо `токенізація`
# не траплялася в корпусі, її триграми можуть перетинатися з триграмами слова
# `токенізатор`. Отже, замість одного беззмістовного `<unk>` ми отримуємо
# часткову інформацію про невідоме слово.

# %%

def char_ngrams(word, n=3):
    """ділить слово на фрагменти по n символів"""
    if n <= 0:
        raise ValueError("n має бути більше нуля")
    if not word:
        return []
    marked = "^" + word + "$"
    return [marked[i:i + n] for i in range(len(marked) - n + 1)]


def ngram_overlap(word_a, word_b, n=3):
    """порівнює множини n-грам"""
    a = set(char_ngrams(word_a, n))
    b = set(char_ngrams(word_b, n))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def test_ngrams():
    assert char_ngrams("кіт", 3) == ["^кі", "кіт", "іт$"]
    assert char_ngrams("кіт", 2) == ["^к", "кі", "іт", "т$"]
    assert char_ngrams("", 3) == []
    assert ngram_overlap("токенізатор", "токенізація", 3) > 0.4
    assert ngram_overlap("кіт", "трактор", 3) == 0.0

    try:
        char_ngrams("слово", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("n має бути додатним")


if __name__ == "__main__":
    test_ngrams()
    print("✓ character n-grams")

# Питання:
# - Які переваги й недоліки матимуть біграми порівняно з п'ятиграмами?
# - Чи зникає OOV повністю, якщо алфавіт під час інференсу теж може змінитися?
# - Чому набір n-грам добре показує схожість, але не зберігає їх порядок?

# %%

def build_ngram_vocab(texts, n=3):
    """збирає відомі n-грами навчального корпусу"""
    grams = set()
    for text in texts:
        for word in word_tokenize(text):
            grams.update(char_ngrams(word, n))
    vocab = {"<pad>": 0, "<unk>": 1}
    for gram in sorted(grams):
        vocab[gram] = len(vocab)
    return vocab


def encode_ngrams(text, vocab, n=3):
    """замінює n-грами тексту на id зі словника"""
    return [vocab.get(gram, vocab["<unk>"])
            for word in word_tokenize(text)
            for gram in char_ngrams(word, n)]


def test_ngram_vocab():
    vocab = build_ngram_vocab(["кіт"], 3)
    assert len(encode_ngrams("кіт", vocab)) == 3
    assert 1 not in encode_ngrams("кіт", vocab)
    assert 1 in encode_ngrams("cat", vocab)
    assert vocab["<pad>"] == 0 and vocab["<unk>"] == 1


if __name__ == "__main__":
    test_ngram_vocab()
    print("✓ n-gram vocabulary")

# %% [markdown]
# ## 3. Byte Pair Encoding (BPE)
#
# Символьна токенізація не має OOV для знайомого алфавіту, але створює довгі
# послідовності. Токенізація словами дає короткі послідовності, але величезний
# словник та OOV. **BPE** шукає компроміс:
#
# 1. починаємо з окремих символів;
# 2. рахуємо частоти всіх сусідніх пар у корпусі;
# 3. найчастішу пару об'єднуємо в один токен;
# 4. повторюємо задану кількість разів.
#
# Часті фрагменти стають одним токеном, а рідкі слова все одно можна скласти з
# менших частин. Маркер `</w>` позначає кінець слова й не дає зливати символи
# з різних слів.

# %%

def word_frequencies(texts):
    """рахує слова без чисел і пунктуації"""
    from collections import Counter
    counts = Counter()
    for text in texts:
        for token in word_tokenize(text):
            if token[0].isalpha():
                counts[token] += 1
    return dict(counts)


def most_frequent_pair(vocabulary):
    """шукає найчастішу пару, враховуючи повторення слів"""
    from collections import Counter
    pairs = Counter()
    for symbols, count in vocabulary.items():
        for pair in zip(symbols, symbols[1:]):
            pairs[pair] += count
    if not pairs:
        return None
    return min(pairs, key=lambda pair: (-pairs[pair], pair))


def merge_pair(symbols, pair):
    """зливає сусідні елементи вибраної пари"""
    result = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            result.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return tuple(result)


def _training_counts(vocabulary):
    """зберігає частоти і список слів, у яких є кожна пара"""
    from collections import Counter, defaultdict
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
    return words, counts, pair_counts, pair_words, symbol_counts


def _merge_training_pair(state, pair, merge):
    """оновлює частоти тільки у словах із вибраною парою"""
    from collections import Counter
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
    """навчає послідовність злиттів BPE"""
    if num_merges < 0:
        raise ValueError("кількість злиттів не може бути від’ємною")
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
    """застосовує злиття в тому порядку, як їх вивчили"""
    if not word:
        return []
    symbols = tuple(word) + ("</w>",)
    for pair in merges:
        symbols = merge_pair(symbols, pair)
    return list(symbols)


def test_bpe():
    vocabulary = {
        ("м", "а", "м", "а", "</w>"): 2,
        ("м", "а", "л", "а", "</w>"): 1,
    }
    assert most_frequent_pair(vocabulary) == ("м", "а")
    assert merge_pair(("м", "а", "м", "а", "</w>"), ("м", "а")) == (
        "ма", "ма", "</w>"
    )

    # Усі символи слова "мамонт" є в train-корпусі, але самого слова немає.
    merges = train_bpe(["мама мама мала тон"], 3)
    assert merges[0] == ("м", "а")
    assert apply_bpe("мама", merges) == ["мама</w>"]

    # Незнайоме слово теж розкладається: воно не стає одним <unk>.
    unseen = apply_bpe("мамонт", merges)
    assert "<unk>" not in unseen
    assert "".join(unseen).replace("</w>", "") == "мамонт"


if __name__ == "__main__":
    test_bpe()
    print("✓ BPE")

# %% [markdown]
# ### Чи справді BPE скорочує послідовності?
#
# Перевіримо це вимірюванням, а не припущенням. До навчання одне слово має
# приблизно стільки токенів, скільки в ньому символів. Після кожного злиття
# деякі часті пари займають одну позицію замість двох.
#
# Важливо: більше злиттів зазвичай означає коротші послідовності, але більший
# словник. BPE не гарантує, що кожне окреме слово стане коротшим, особливо якщо
# воно не схоже на навчальний корпус.

# %%

def average_bpe_length(texts, merges):
    """рахує середню довжину слова в BPE-токенах"""
    frequencies = word_frequencies(texts)
    total_words = sum(frequencies.values())
    if total_words == 0:
        raise ValueError("у корпусі немає слів")
    total_tokens = sum(len(apply_bpe(word, merges)) * count
                       for word, count in frequencies.items())
    return total_tokens / total_words


def test_average_bpe_length():
    corpus = [
        "токен токени токенізація",
        "токенізатор токенізує текст",
    ]
    merges = train_bpe(corpus, 20)
    before = average_bpe_length(corpus, [])
    after = average_bpe_length(corpus, merges)
    assert after < before
    assert before > 1


if __name__ == "__main__":
    test_average_bpe_length()
    print("✓ BPE reduces average sequence length")

# Питання:
# - Що ми отримуємо ціною збільшення словника BPE?
# - Чому коректніше порівнювати довжину на окремому test-корпусі?
# - Звичайний BPE починається із символів. Як byte-level BPE уникає OOV навіть
#   для емодзі, нового алфавіту або рідкісного Unicode-символу?

# %% [markdown]
# ## 4. WordPiece
#
# WordPiece навчається майже як BPE: починає із символів і поступово зливає
# сусідні фрагменти. Значну частину логіки можна адаптувати з попереднього завдання.
#
# Відмінності: `##` позначає продовження слова; пари вибираються за оцінкою,
# а не лише за частотою; під час токенізації шукають найдовший фрагмент зі словника.
#
# `score(a, b) = frequency(a, b) / (frequency(a) * frequency(b))`

# %%


def wordpiece_symbols(word):
    """додає ## до символів, які продовжують слово"""
    if not word:
        return ()
    return (word[0],) + tuple("##" + char for char in word[1:])


def wordpiece_pair_scores(vocabulary):
    """рахує оцінку пари за частотами її частин"""
    from collections import Counter
    symbols_count = Counter()
    pairs_count = Counter()
    for symbols, count in vocabulary.items():
        for symbol in symbols:
            symbols_count[symbol] += count
        for pair in zip(symbols, symbols[1:]):
            pairs_count[pair] += count
    return {pair: count / (symbols_count[pair[0]] * symbols_count[pair[1]])
            for pair, count in pairs_count.items()}


def merge_wordpiece_pair(symbols, pair):
    """зливає пару і залишає один префікс продовження"""
    result = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            result.append(symbols[i] + symbols[i + 1].removeprefix("##"))
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return tuple(result)


def train_wordpiece(texts, vocab_size):
    """навчає словник WordPiece"""
    if vocab_size <= 0:
        raise ValueError("розмір словника має бути більше нуля")
    vocabulary = {wordpiece_symbols(word): count
                  for word, count in word_frequencies(texts).items()}
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
        pair = min(pair_counts, key=lambda p: (
            -pair_counts[p] / (symbol_counts[p[0]] * symbol_counts[p[1]]), p
        ))
        merged_token = pair[0] + pair[1].removeprefix("##")
        _merge_training_pair(state, pair, merge_wordpiece_pair)
        vocab.add(merged_token)
    return vocab


def apply_wordpiece(word, vocab, unk_token="[UNK]"):
    """на кожній позиції бере найдовший відомий фрагмент"""
    pieces = []
    start = 0
    while start < len(word):
        found = None
        for end in range(len(word), start, -1):
            piece = word[start:end]
            if start > 0:
                piece = "##" + piece
            if piece in vocab:
                found = piece
                start = end
                break
        if found is None:
            return [unk_token]
        pieces.append(found)
    return pieces


def test_wordpiece():
    weighted_vocab = {
        ("а", "##б"): 2,
        ("а", "##в"): 1,
        ("г", "##в"): 2,
    }
    scores = wordpiece_pair_scores(weighted_vocab)
    assert scores[("а", "##б")] > scores[("а", "##в")]
    assert merge_wordpiece_pair(("м", "##а", "##м", "##а"), ("м", "##а")) == (
        "ма", "##м", "##а"
    )

    vocab = train_wordpiece(["мама мама мала тон"], 30)
    assert "[UNK]" in vocab
    assert apply_wordpiece("мама", vocab) == ["мама"]

    # На відміну від BPE, WordPiece шукає найдовший фрагмент зі словника,
    # а не повторює послідовність вивчених злиттів.
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


if __name__ == "__main__":
    test_wordpiece()
    print("✓ WordPiece")

# Питання:
# - Чому `##` потрібен, щоб відрізняти початок слова від його продовження?
# - Чим WordPiece-score відрізняється від вибору найчастішої пари у BPE?
# - Чому BERT повертає `[UNK]` для цілого слова, якщо не знаходить один фрагмент?

# %% [markdown]
# ## 5. Великий експеримент: FLORES та GPT
#
# Тепер застосуємо вивчені ідеї до справжнього багатомовного корпусу. Навчимо
# власні токенізатори на англійській, українській та грузинській мовах, а потім
# порівняємо їх із готовими токенізаторами GPT.
#
# Використаємо `dev` split FLORES-200: по 997 паралельних речень кожною мовою,
# тобто 2991 навчальний рядок. Це небагато для реальної мовної моделі, але
# достатньо для навчального експерименту з токенізацією.
#
# Код завантаження даних уже готовий. Його реалізовувати не потрібно.

# %%

from datasets import load_dataset


FLORES_LANGUAGES = {
    "English": "eng_Latn",
    "Українська": "ukr_Cyrl",
    "ქართული": "kat_Geor",
}


def load_flores_dev():
    """Завантажує по 997 речень FLORES для кожної вибраної мови."""
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


if __name__ == "__main__":
    flores = load_flores_dev()

    assert all(len(texts) == 997 for texts in flores.values())
    train_texts = [
        text
        for language_texts in flores.values()
        for text in language_texts
    ]

    print({language: len(texts) for language, texts in flores.items()})
    print("Усього навчальних рядків:", len(train_texts))

# %% [markdown]
# Подивіться на одне й те саме речення трьома мовами. FLORES є паралельним
# корпусом, тому рядки з однаковим індексом мають однаковий зміст.

# %%

if __name__ == "__main__":
    example_index = 0
    for language, texts in flores.items():
        print(f"{language}: {texts[example_index]}\n")

# Питання:
# - Чому для порівняння важливо мати однакову кількість текстів кожною мовою?
# - Чи є 997 речень репрезентативними для всього різноманіття мови?

# %% [markdown]
# ### Навчання на FLORES із різними словниками
#
# Тут не створюємо нових алгоритмів. Повторно використайте вже реалізовані
# `train_bpe`, `apply_bpe`, `train_wordpiece` та `apply_wordpiece`.
#
# Навчіть обидва токенізатори зі словниками приблизно 500, 1000 і 2000 токенів.
# Для BPE початковий словник уже містить символи, а кожне merge додає один
# токен, тому бажаний розмір словника треба перетворити на кількість merges.

# %%

VOCAB_SIZES = (500, 1000, 2000)


def bpe_num_merges(texts, vocab_size):
    """Обчислює кількість BPE-merges для бажаного розміру словника."""
    frequencies = word_frequencies(texts)
    characters = {
        character
        for word in frequencies
        for character in word
    }
    initial_vocab_size = len(characters) + 1  # </w>
    if vocab_size < initial_vocab_size:
        raise ValueError("vocab_size менший за початковий алфавіт")
    return vocab_size - initial_vocab_size


def apply_bpe_to_text(text, merges):
    """Застосовує створений раніше BPE до всіх токенів тексту."""
    return [
        piece
        for token in word_tokenize(text)
        for piece in apply_bpe(token, merges)
    ]


def apply_wordpiece_to_text(text, vocab):
    """Застосовує створений раніше WordPiece до всіх токенів тексту."""
    return [
        piece
        for token in word_tokenize(text)
        for piece in apply_wordpiece(token, vocab)
    ]


if __name__ == "__main__":
    bpe_models = {}
    wordpiece_models = {}

    for vocab_size in VOCAB_SIZES:
        print("Навчаю словники:", vocab_size, flush=True)
        bpe_models[vocab_size] = train_bpe(
            train_texts,
            bpe_num_merges(train_texts, vocab_size),
        )
        wordpiece_models[vocab_size] = train_wordpiece(
            train_texts,
            vocab_size,
        )

    print("Навчено BPE і WordPiece для:", VOCAB_SIZES)

# Питання:
# - Як змінюється час навчання зі збільшенням словника?
# - Чи розподіляються токени порівну між трьома системами письма?
# - Чи завжди більший словник дає нижчу fertility?

# %% [markdown]
# ### Готові GPT-токенізатори
#
# Наступний код уже готовий. `tiktoken` імпортує ті самі encoding-и, які
# використовують відповідні покоління GPT. GPT-3.5 і класичний GPT-4 мають
# однаковий `cl100k_base`, тому їхні результати збігатимуться.

# %%

import tiktoken


if __name__ == "__main__":
    gpt_encodings = {
        "GPT-2": tiktoken.encoding_for_model("gpt-2"),
        "GPT-3.5": tiktoken.encoding_for_model("gpt-3.5-turbo"),
        "GPT-4": tiktoken.encoding_for_model("gpt-4"),
        "GPT-5": tiktoken.encoding_for_model("gpt-5"),
    }

    for name, encoding in gpt_encodings.items():
        print(f"{name:7} -> {encoding.name:12} ({encoding.n_vocab} tokens)")

# %% [markdown]
# ### Fertility
#
# **Fertility** — середня кількість токенів на одне слово:
#
# $$\text{fertility} =
# \frac{\text{загальна кількість токенів}}
#      {\text{загальна кількість слів}}$$
#
# У цій лабораторній словом вважаємо непорожній фрагмент після `str.split()`.
# Рахуйте відношення двох сум для всього корпусу, а не середнє значень окремих
# речень: тоді кожне слово має однакову вагу.

# %%

def fertility_table(tokenizers, corpora):
    """збирає fertility для кожного токенізатора і мови"""
    results = {}
    for name, tokenize in tokenizers.items():
        results[name] = {}
        for language, texts in corpora.items():
            results[name][language] = fertility(tokenize, texts)
    return results


def cached_subword_tokenizer(tokenize_word):
    """повторно використовує готове розбиття знайомих слів"""
    from functools import lru_cache
    cached_word = lru_cache(maxsize=50000)(tokenize_word)

    def tokenize(text):
        return [piece for word in word_tokenize(text)
                for piece in cached_word(word)]
    return tokenize


def fertility(tokenize, texts):
    """ділить загальну кількість токенів на кількість слів"""
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


if __name__ == "__main__":
    test_fertility()
    print("✓ fertility")

# %% [markdown]
# ### Порівняння за мовами
#
# Побудуйте таблицю fertility для всіх 10 токенізаторів:
#
# - BPE: 500, 1000, 2000;
# - WordPiece: 500, 1000, 2000;
# - GPT-2, GPT-3.5, GPT-4, GPT-5.
#
# Наші моделі перевіряються на тому самому `dev`, на якому навчалися. Отже, це
# train fertility — вона показує стиснення корпусу, але не узагальнення.

# %%

if __name__ == "__main__":
    tokenizers_to_compare = {}

    for vocab_size, merges in bpe_models.items():
        tokenizers_to_compare[f"BPE-{vocab_size}"] = cached_subword_tokenizer(
            lambda word, merges=merges: apply_bpe(word, merges)
        )

    for vocab_size, vocab in wordpiece_models.items():
        tokenizers_to_compare[f"WordPiece-{vocab_size}"] = cached_subword_tokenizer(
            lambda word, vocab=vocab: apply_wordpiece(word, vocab)
        )

    for name, encoding in gpt_encodings.items():
        tokenizers_to_compare[name] = encoding.encode




    results = fertility_table(tokenizers_to_compare, flores)

    header = f"{'tokenizer':<24}" + "".join(
        f"{language:>14}" for language in flores
    )
    print(header)
    print("-" * len(header))
    for tokenizer_name, language_values in results.items():
        row = f"{tokenizer_name:<24}" + "".join(
            f"{language_values[language]:14.3f}" for language in flores
        )
        print(row)

# Питання:
# - Для якої мови GPT-2 має найбільшу fertility? Чому?
# - Як змінюється fertility при переході 500 → 1000 → 2000?
# - Чому GPT-токенізатор із великим словником може програти маленькому
#   токенізатору, навченому саме на цих трьох мовах?

# %% [markdown]
# ### Змішані та складні випадки
#
# Середня fertility приховує конкретні невдалі розбиття. Порівняйте кількість
# та самі токени для code-switching, різних систем письма, чисел, emoji, URL і
# слів, яких не було у FLORES.

# %%

MIXED_CASES = [
    "Я навчаю tokenizer на FLORES dataset.",
    "The model розуміє ქართულ ენას?",
    "Київ -> Tbilisi -> თბილისი -> London 🌍",
    "Замовлення №2026-09-24 коштує $19.99.",
    "support@example.com або https://example.com/допомога",
    "я дуже втомився від курсу основи Large Language Models"
]


def compare_mixed_cases(texts, tokenizers):
    """показує кількість токенів у змішаних реченнях"""
    for index, text in enumerate(texts, 1):
        print(f"{index}. {text}")
    print()
    print(f"{'tokenizer':<18}" + "".join(f"{i:>7}" for i in range(1, len(texts) + 1)))
    for name, tokenize in tokenizers.items():
        counts = [len(tokenize(text)) for text in texts]
        print(f"{name:<18}" + "".join(f"{count:>7}" for count in counts))


if __name__ == "__main__":
    compare_mixed_cases(MIXED_CASES, tokenizers_to_compare)

# Фінальні питання:
# - Які символи або фрагменти стабільно створюють найбільше токенів?
# - Який tokenizer найкраще поводиться, коли в одному реченні три мови?
# - Чому fertility недостатньо для остаточного вибору токенізатора?

# %% [markdown]
# ## 6. Послівний словник і триграми на FLORES
#
# Я також зібрав словники слів і триграм на цих самих 2991 реченнях.
# Це train-перевірка, тому низька частка невідомих токенів тут очікувана.

# %%

if __name__ == "__main__":
    word_vocab = build_word_vocab(train_texts)
    trigram_vocab = build_ngram_vocab(train_texts, 3)
    baselines = {
        "Words": lambda text: encode_words(text, word_vocab),
        "Trigrams": lambda text: encode_ngrams(text, trigram_vocab, 3),
    }
    print("Словник слів:", len(word_vocab))
    print("Словник триграм:", len(trigram_vocab))
    print(f"{'tokenizer':<12} {'language':<14} {'fertility':>10} {'UNK, %':>9}")
    for name, tokenize_text in baselines.items():
        for language, texts in flores.items():
            ids = [token for text in texts for token in tokenize_text(text)]
            unknown_rate = 100 * ids.count(1) / len(ids) if ids else 0
            print(f"{name:<12} {language:<14} {fertility(tokenize_text, texts):10.3f} {unknown_rate:9.2f}")

# %% [markdown]
# ## 7. Розміри словників і невідомі токени
#
# Самої fertility недостатньо: [UNK] теж займає одну позицію, але втрачає слово.
# Нижче я рахую частку вхідних токенів word_tokenize, які WordPiece замінив на [UNK].
# До них входять і числа та пунктуація: за умовою навчання використало тільки слова.

# %%

if __name__ == "__main__":
    alphabet = set("".join(word_frequencies(train_texts))) | {"</w>"}
    for size, merges in bpe_models.items():
        learned = alphabet | {a + b for a, b in merges}
        print(f"BPE-{size}: {len(learned)} токенів, {len(merges)} злиттів")
    for size, vocab in wordpiece_models.items():
        print(f"WordPiece-{size}: {len(vocab)} токенів")
    print()
    for language, texts in flores.items():
        words = [word for text in texts for word in word_tokenize(text)]
        rates = []
        for size, vocab in wordpiece_models.items():
            unknown = sum(apply_wordpiece(word, vocab) == ["[UNK]"] for word in words)
            rates.append(f"WP-{size}: {100 * unknown / len(words):.2f}%")
        print(language, "; ".join(rates))

# %% [markdown]
# ## 8. Порівняння fertility на графіку
#
# Нижче значення з тієї самої таблиці. Менше токенів на слово означає коротшу
# послідовність, але не гарантує кращого збереження тексту або якості моделі.

# %%

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(results)
    languages = list(flores)
    values = np.array([[results[name][lang] for lang in languages] for name in names])
    fig, ax = plt.subplots(figsize=(7, 5.4), layout="constrained")
    image = ax.imshow(values, cmap="YlOrRd", vmin=0, aspect="auto")
    ax.set_xticks(range(3), ["English", "Ukrainian", "Georgian"])
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Fertility на FLORES dev: 997 речень на мову", pad=12)
    for i in range(len(names)):
        for j in range(3):
            color = "white" if values[i, j] > values.max() * 0.62 else "black"
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, label="токенів на слово", shrink=0.85)
    plt.show()

# %% [markdown]
# ## 9. Приклади самого розбиття
#
# Порівнюю злиття, ## та поведінку на emoji. GPT працює з байтами, тому окремий
# токен не завжди декодується як цілий Unicode-символ. У такому разі показую hex.

# %%

if __name__ == "__main__":
    samples = ["Я навчаю tokenizer на FLORES dataset.", "Київ თბილისი 🌍"]
    for text in samples:
        print(text)
        for name in ("BPE-2000", "WordPiece-2000", "GPT-5"):
            tokens = tokenizers_to_compare[name](text)
            if name == "GPT-5":
                pieces = []
                for token in tokens:
                    raw = gpt_encodings[name].decode_single_token_bytes(token)
                    try:
                        pieces.append(raw.decode("utf-8"))
                    except UnicodeDecodeError:
                        pieces.append("0x" + raw.hex())
            else:
                pieces = tokens
            print(f"{name} ({len(tokens)}): {pieces}")
        for name, tokenize_text in baselines.items():
            ids = tokenize_text(text)
            print(f"{name}: {len(ids)} токенів, {ids.count(1)} невідомих")
        print()

# %% [markdown]
# ## Висновок
#
# Я реалізував усі функції з шаблону та перевірив їх тестами. Словник слів має 21352
# елементи, словник триграм - 15649. На навчальному корпусі в них немає невідомих токенів,
# але в нових змішаних прикладах вони вже з’являються.
#
# Для BPE збільшення словника з 500 до 2000 скоротило fertility: англійська
# 3.387 → 2.469, українська 4.323 → 3.185, грузинська 4.928 → 3.565.
# У WordPiece скорочення слабше. Його [UNK] замінив 13.22%, 18.24% і 19.01% вхідних
# токенів word_tokenize відповідно. Серед них є числа та пунктуація, бо навчання за
# умовою брало тільки слова. Через це коротша послідовність не завжди краща.
#
# GPT-2 добре стискає англійський текст (1.228), але грузинський має fertility 21.545.
# GPT-3.5 і GPT-4 дали однакові числа, бо використовують cl100k_base.
# Найнижчі значення серед десяти порівняних варіантів у цьому досліді має GPT-5:
# 1.219 для англійської, 2.442 для української, 2.780 для грузинської.
#
# Більший словник потребує більше кроків навчання. Токени не зобов’язані розподілятися
# порівну між мовами: однакова кількість речень не означає однакові частоти фрагментів.
# Зниження fertility не гарантоване для кожного нового тексту.
#
# Це порівняння на невеликому FLORES dev, де власні моделі також навчалися.
# Наші токенізатори знижують регістр, а GPT зберігає його. Тому результати не доводять
# загальну перевагу алгоритму. Для вибору токенізатора потрібні також окремий тестовий
# корпус, частка невідомих слів, збереження тексту, швидкість і поведінка на різних мовах.
