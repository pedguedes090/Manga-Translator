import pytest
from translator import MangaTranslator


JA_TEXT = "こんばんわ!"
EN_TRANSLATION = "good evening!"


@pytest.fixture
def translator():
    return MangaTranslator()


@pytest.mark.parametrize("method", ["google", "hf", "baidu", "bing"])
def test_translate(translator, method):
    translated_text = translator.translate(JA_TEXT, method=method)
    assert translated_text.lower() == EN_TRANSLATION


def test_invalid_translation_method(translator):
    with pytest.raises(ValueError) as e:
        translator.translate(JA_TEXT, method="Mirai")
    assert str(e.value) == "Invalid translation method."
