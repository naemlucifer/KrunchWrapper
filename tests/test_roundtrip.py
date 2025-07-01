from core.compress import compress, decompress


def test_roundtrip():
    sample = "def hello(name):\n    print('Hello', name)\n"
    lang = "python"

    packed = compress(sample, lang)
    assert packed.text != ""  # something returned

    restored = decompress(packed.text, packed.used)
    assert restored == sample 