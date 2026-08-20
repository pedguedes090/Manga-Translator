import importlib
import json
import os
import builtins
import uuid

import cv2
import numpy as np
import pytest

from add_text import (
    _compute_font_and_wrap,
    _decide_skip_render,
    _filter_components_outside_inner,
    assess_erasability,
    erase_text_region,
    merge_nearby_ocr_blocks,
    refine_tall_narrow_ocr_bbox,
    should_skip_ocr_artifact,
    sort_ocr_blocks_reading_order,
)
from translator import MangaTranslator


translator_module = importlib.import_module("translator.translator")


class FakeGoogleTranslator:
    created = []

    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.created.append((source, target))

    def translate(self, text):
        return f"{self.source}->{self.target}:{text}"


@pytest.fixture(autouse=True)
def fake_google(monkeypatch):
    FakeGoogleTranslator.created = []
    monkeypatch.setattr(translator_module, "GoogleTranslator", FakeGoogleTranslator)


def test_google_translator_maps_chinese_code():
    translator = MangaTranslator(source="zh", target="vi")

    translated_text = translator.translate("你好", method="google")

    assert translated_text == "zh-CN->vi:你好"
    assert FakeGoogleTranslator.created == [("zh-CN", "vi")]


def test_google_batch_keeps_empty_text_and_order():
    translator = MangaTranslator(source="zh", target="en")

    translated = translator.translate_batch_google(["你好", "", "再见"])

    assert translated == ["zh-CN->en:你好", "", "zh-CN->en:再见"]


def test_preprocess_ocr_line_breaks_inside_cjk_words():
    translator = MangaTranslator(source="zh", target="vi")

    cleaned = translator._preprocess_text("都这种时候了,\n南夏会让傲之\n柱看些什么?")

    assert cleaned == "都这种时候了, 南夏会让傲之柱看些什么?"


def test_gemini_batch_rotates_to_next_key_when_current_key_is_exhausted():
    from translator.gemini_translator import GeminiTranslator

    calls = []

    class FakeModels:
        def __init__(self, api_key):
            self.api_key = api_key

        def generate_content(self, model, contents):
            calls.append(self.api_key)
            if self.api_key == "bad-key":
                raise RuntimeError("429 quota exceeded")

            class Response:
                text = '["xin chao", "tam biet"]'

            return Response()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels(api_key)

    translator = GeminiTranslator(
        api_keys=["bad-key", "good-key"],
        client_factory=lambda api_key: FakeClient(api_key),
    )

    translated = translator.translate_batch(["hello", "bye"], source="en", target="vi")

    assert translated == ["xin chao", "tam biet"]
    assert calls == ["bad-key", "good-key"]


def test_gemini_uses_configured_model_name():
    from translator.gemini_translator import GeminiTranslator

    called_models = []

    class FakeModels:
        def generate_content(self, model, contents):
            called_models.append(model)

            class Response:
                text = '["xin chao"]'

            return Response()

    class FakeClient:
        models = FakeModels()

    translator = GeminiTranslator(
        api_keys=["key"],
        model_name="gemini-custom-model",
        client_factory=lambda api_key: FakeClient(),
    )

    assert translator.translate_batch(["hello"], source="en", target="vi") == ["xin chao"]
    assert called_models == ["gemini-custom-model"]


def test_gemini_default_model_is_3_1_flash_lite():
    from translator.gemini_translator import GeminiTranslator

    translator = GeminiTranslator(api_keys=["key"])

    assert translator.model_name == "gemini-3.1-flash-lite"


def test_gemini_batch_returns_original_texts_after_all_keys_fail():
    from translator.gemini_translator import GeminiTranslator

    calls = []

    class FakeModels:
        def __init__(self, api_key):
            self.api_key = api_key

        def generate_content(self, model, contents):
            calls.append(self.api_key)
            raise RuntimeError("API key not valid")

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels(api_key)

    translator = GeminiTranslator(
        api_keys=["bad-1", "bad-2"],
        client_factory=lambda api_key: FakeClient(api_key),
    )

    translated = translator.translate_batch(["hello"], source="en", target="vi")

    assert translated == ["hello"]
    assert calls == ["bad-1", "bad-2"]
    assert translator.exhausted_api_keys == {"bad-1", "bad-2"}


def test_local_llm_retries_the_whole_batch_when_one_translation_is_missing():
    from translator.local_llm_translator import LocalLLMTranslator

    class RepairingLocalLLM(LocalLLMTranslator):
        def __init__(self):
            super().__init__()
            self.batch_calls = 0

        def _post_chat(self, prompt, timeout):
            self.batch_calls += 1
            count = 18 if self.batch_calls == 1 else 19
            return json.dumps([f"dịch {index}" for index in range(count)])

        def translate_single(self, text, source="ja", target="en"):
            raise AssertionError("a malformed batch must not lose shared context")

    translator = RepairingLocalLLM()
    texts = [f"source {index}" for index in range(19)]

    translated = translator.translate_batch(texts, source="ko", target="vi")

    assert translated == [f"dịch {index}" for index in range(19)]
    assert translator.batch_calls == 2


def test_local_llm_maps_indexed_batch_response_back_to_input_order():
    from translator.local_llm_translator import LocalLLMTranslator

    class IndexedLocalLLM(LocalLLMTranslator):
        def _post_chat(self, prompt, timeout):
            return json.dumps(
                [
                    {"id": 2, "translation": "ba"},
                    {"id": 0, "translation": "một"},
                    {"id": 1, "translation": "hai"},
                ]
            )

    translator = IndexedLocalLLM()

    translated = translator.translate_batch(
        ["one", "two", "three"], source="en", target="vi"
    )

    assert translated == ["một", "hai", "ba"]


def test_gemini_default_client_factory_requires_google_genai(monkeypatch):
    from translator.gemini_translator import GeminiTranslator

    real_import = builtins.__import__

    def fail_google_genai_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google" and "genai" in fromlist:
            raise ImportError("google-genai missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_google_genai_import)
    translator = GeminiTranslator(api_keys=["fake-key"])

    with pytest.raises(RuntimeError, match="google-genai SDK is required"):
        translator._get_client("fake-key")


def test_parse_gemini_api_keys_accepts_newlines_commas_and_semicolons():
    from app import parse_gemini_api_keys

    keys = parse_gemini_api_keys(" key-1\nkey-2, key-3;key-2 ")

    assert keys == ["key-1", "key-2", "key-3"]


def test_build_result_images_includes_original_image_data_for_compare():
    from app import build_result_images

    original = np.full((16, 16, 3), 255, dtype=np.uint8)
    processed = np.zeros((16, 16, 3), dtype=np.uint8)

    images = build_result_images(
        [{"name": "page_1", "image": processed}],
        {"page_1": original},
    )

    assert len(images) == 1
    assert images[0]["name"] == "page_1"
    assert images[0]["data"]
    assert images[0]["original_data"]
    assert images[0]["data"] != images[0]["original_data"]


def test_snapshot_original_images_keeps_independent_copies_for_compare():
    from app import snapshot_original_images

    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    originals = snapshot_original_images([("page", image, [])])

    image[:, :] = 0

    assert np.all(originals["page"] == 100)


def test_upload_helpers_allow_only_supported_images_and_clean_names():
    from app import clean_image_name, is_allowed_image_file

    assert is_allowed_image_file("chapter-01.PNG")
    assert is_allowed_image_file("page.jpeg")
    assert is_allowed_image_file("cover.webp")
    assert is_allowed_image_file("scan.tiff")
    assert is_allowed_image_file("panel.bmp")
    assert is_allowed_image_file("new-format.avif")
    assert not is_allowed_image_file("notes.txt")
    assert not is_allowed_image_file("")
    assert clean_image_name("../../chapter 01.png") == "chapter_01"
    assert clean_image_name("???.jpg") == "image"


def test_translate_and_render_records_warning_when_local_llm_falls_back(monkeypatch):
    import app as app_module

    class BrokenLocalLLM:
        def translate_batch(self, texts, source, target):
            raise RuntimeError("server down")

        def translate_single(self, text, source, target):
            raise RuntimeError("still down")

    class TranslatorObj:
        _local_llm_tr = BrokenLocalLLM()

    monkeypatch.setattr(
        app_module,
        "erase_text_region",
        lambda image, bbox, source_lang="en": (image, (0, 0, 0), {}),
    )
    monkeypatch.setattr(app_module, "render_all_blocks", lambda image, blocks, font_path: image)

    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    blocks = [{"text": "hello", "bbox": [2, 2, 28, 20]}]
    translator = TranslatorObj()

    app_module.translate_and_render(
        [("page", image, blocks)],
        translator,
        "animeace_",
        "copilot",
        "en",
        "vi",
        "",
    )

    assert "Local LLM" in translator.last_warning


def test_translate_and_render_uses_configured_gemini_translator(monkeypatch):
    import app as app_module

    class FakeGeminiTranslator:
        def __init__(self):
            self.batch_calls = []

        def translate_batch(self, texts, source, target):
            self.batch_calls.append((texts, source, target))
            return ["xin chao"]

        def translate_single(self, text, source, target):
            raise AssertionError("Gemini batch translation should be used first")

    class TranslatorObj:
        def __init__(self):
            self._gemini_translator = FakeGeminiTranslator()

    monkeypatch.setattr(
        app_module,
        "erase_text_region",
        lambda image, bbox, source_lang="en": (image, (0, 0, 0), {}),
    )
    monkeypatch.setattr(app_module, "render_all_blocks", lambda image, blocks, font_path: image)

    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    blocks = [{"text": "hello", "bbox": [2, 2, 28, 20]}]
    translator = TranslatorObj()

    app_module.translate_and_render(
        [("page", image, blocks)],
        translator,
        "animeace_",
        "gemini",
        "en",
        "vi",
        "",
    )

    assert translator._gemini_translator.batch_calls == [(["hello"], "en", "vi")]


def test_session_roundtrip_preserves_expanded_bbox_metadata(monkeypatch, tmp_path):
    import app as app_module

    session_id = str(uuid.uuid4())
    monkeypatch.setattr(app_module, "TEMP_DIR", str(tmp_path))
    app_module.ocr_sessions.clear()

    image = np.full((48, 48, 3), 255, dtype=np.uint8)
    session_data = {
        "all_ocr_results": [
            (
                "page",
                image,
                [
                    {
                        "text": "hello",
                        "bbox": [10, 10, 20, 20],
                        "_bbox_expanded": True,
                        "_scaled_retry": True,
                        "_erasability": {"reason": "uniform_background", "score": 1.0},
                    }
                ],
            )
        ],
    }

    app_module._save_session(session_id, session_data)
    app_module.ocr_sessions.clear()

    loaded = app_module.load_session(session_id)
    block = loaded["all_ocr_results"][0][2][0]
    preview = app_module.build_preview_images(loaded["all_ocr_results"])

    assert block["_bbox_expanded"] is True
    assert block["_scaled_retry"] is True
    assert block["_erasability"]["reason"] == "uniform_background"
    assert preview[0]["blocks"][0]["bbox"] == [10, 10, 20, 20]


def test_continue_translate_keeps_correction_bboxes_exact(monkeypatch):
    import app as app_module

    session_id = str(uuid.uuid4())
    app_module.ocr_sessions.clear()
    app_module.ocr_sessions[session_id] = {
        "all_ocr_results": [("page", np.full((64, 64, 3), 255, dtype=np.uint8), [])],
        "selected_translator": "google",
        "selected_font": "animeace_",
        "source_lang": "ko",
        "target_lang": "vi",
        "style": "",
        "gemini_api_keys": [],
        "gemini_api_key": "",
        "gemini_model": "gemini-session-model",
        "copilot_server": "",
        "copilot_model": "",
    }
    captured = {}

    def fake_full_pipeline(all_images, all_ocr_results, *args, **kwargs):
        captured["all_ocr_results"] = all_ocr_results
        captured["gemini_model"] = args[7]
        return "OK"

    monkeypatch.setattr(app_module, "_do_full_pipeline", fake_full_pipeline)

    response = app_module.app.test_client().post(
        "/continue-translate",
        data={
            "session_id": session_id,
            "modified_blocks": json.dumps(
                [{
                    "image_idx": 0,
                    "blocks": [
                        {"text": "right", "bbox": [35, 5, 55, 25]},
                        {"text": "bottom", "bbox": [5, 40, 25, 60]},
                        {"text": "left", "bbox": [5, 10, 25, 30]},
                    ],
                }]
            ),
        },
    )

    assert response.status_code == 200
    captured_blocks = captured["all_ocr_results"][0][2]
    assert [block["text"] for block in captured_blocks] == ["left", "right", "bottom"]
    assert [block["_text_idx"] for block in captured_blocks] == [0, 1, 2]
    assert captured_blocks[0]["bbox"] == [5, 10, 25, 30]
    assert captured["gemini_model"] == "gemini-session-model"


def test_ocr_region_joins_blocks_in_reading_order(monkeypatch):
    import app as app_module

    session_id = str(uuid.uuid4())
    app_module.ocr_sessions.clear()
    app_module.ocr_sessions[session_id] = {
        "all_ocr_results": [("page", np.full((80, 80, 3), 255, dtype=np.uint8), [])],
        "source_lang": "en",
    }

    class FakeOCR:
        def __init__(self, ocr_language):
            assert ocr_language == "en"

        def __call__(self, image):
            return [
                {"text": "right", "bbox": [30, 2, 50, 22]},
                {"text": "bottom", "bbox": [2, 35, 25, 55]},
                {"text": "left", "bbox": [2, 6, 22, 26]},
            ]

    monkeypatch.setattr(app_module, "ChromeLensOCR", FakeOCR)
    response = app_module.app.test_client().post(
        "/ocr-region",
        data={
            "session_id": session_id,
            "image_idx": "0",
            "x1": "0",
            "y1": "0",
            "x2": "70",
            "y2": "70",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["text"] == "left right bottom"


def test_build_preview_images_repairs_saved_hangul_tall_narrow_bbox():
    import app as app_module

    image_path = r"F:\webtruyen-main\webtruyen-main\server\uploads\chapters\1\3\027.webp"
    if not os.path.exists(image_path):
        pytest.skip("local user sample image is not available")
    image = cv2.imread(image_path)

    preview = app_module.build_preview_images(
        [
            (
                "027",
                image,
                [
                    {
                        "text": "어머~",
                        "bbox": [115, 1001, 164, 1139],
                        "_bbox_expanded": True,
                    }
                ],
            )
        ],
        source_lang="zh",
    )

    bbox = preview[0]["blocks"][0]["bbox"]
    assert bbox[0] <= 85
    assert bbox[2] >= 200
    assert bbox[1] >= 1035
    assert bbox[3] <= 1110


def test_invalid_translation_method():
    translator = MangaTranslator()
    with pytest.raises(ValueError) as e:
        translator.translate("こんばんわ!", method="Mirai")
    assert str(e.value) == "Invalid translation method."


def test_skip_cjk_page_latin_decoration_and_watermark():
    image_shape = (1136, 800, 3)

    assert should_skip_ocr_artifact("S", [349, 112, 440, 203], image_shape, "zh")
    assert should_skip_ocr_artifact(
        "manhuaren.com", [706, 1020, 800, 1036], image_shape, "zh"
    )


def test_keep_real_cjk_dialogue():
    assert not should_skip_ocr_artifact(
        "都这种时候了,\n南夏会让傲之\n柱看些什么?",
        [245, 310, 401, 390],
        (1136, 800, 3),
        "zh",
    )


def test_render_wrap_normalizes_llm_newlines():
    result = _compute_font_and_wrap(
        "Cực Lạc Băng Hạch ư?\nNam Hạ dùng nó\nđể khởi động ảo ảnh!",
        [0, 0, 130, 80],
        "fonts/animeace_i.ttf",
    )

    assert result is not None
    _, lines, _ = result
    assert all("\n" not in line for line in lines)


def test_erase_text_region_does_not_fill_entire_bbox():
    image = np.full((90, 150, 3), 245, dtype=np.uint8)
    cv2.putText(image, "TEST", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    original_corner = image[22, 12].copy()

    erased, _, appearance = erase_text_region(image, [10, 20, 120, 65], source_lang="en")

    assert appearance["erase_method"] in {
        "stroke-fill",
        "stroke-fill-sampled",
        "stroke-inpaint",
    }
    assert np.array_equal(erased[22, 12], original_corner)


def test_erase_text_region_keeps_bubble_border_intact():
    image = np.full((140, 180, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (90, 70), (65, 42), 0, 0, 360, (0, 0, 0), 3)
    cv2.putText(image, "HEY", (52, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    border_pixel = image[70, 25].copy()

    erased, _, appearance = erase_text_region(image, [48, 48, 125, 82], source_lang="en")

    assert appearance["erase_method"] in {"stroke-fill", "stroke-fill-sampled", "stroke-inpaint"}
    assert np.array_equal(erased[70, 25], border_pixel)


def test_erase_text_region_keeps_border_when_bbox_overlaps_it():
    image = np.full((140, 180, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (90, 70), (65, 42), 0, 0, 360, (0, 0, 0), 3)
    cv2.putText(image, "HEY", (52, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    border_pixel = image[70, 25].copy()

    erased, _, appearance = erase_text_region(image, [24, 45, 125, 85], source_lang="en")

    assert appearance["erase_method"] in {"stroke-fill", "stroke-fill-sampled", "stroke-inpaint"}
    assert np.array_equal(erased[70, 25], border_pixel)


def test_erase_text_region_removes_white_outline_on_gray_title_card():
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 35), (220, 125), (204, 204, 204), -1)
    cv2.putText(image, "Q. TITLE", (32, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5)
    cv2.putText(image, "Q. TITLE", (32, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    erased, _, _ = erase_text_region(image, [25, 58, 215, 92], source_lang="ko")
    gray = cv2.cvtColor(erased[58:92, 25:215], cv2.COLOR_BGR2GRAY)

    assert np.count_nonzero(gray > 230) < 80


def test_filter_components_keeps_text_cropped_by_ocr_edge_but_drops_thin_border():
    mask = np.zeros((40, 80), dtype=np.uint8)
    cv2.rectangle(mask, (12, 0), (29, 39), 255, -1)
    cv2.rectangle(mask, (77, 0), (79, 39), 255, -1)

    filtered = _filter_components_outside_inner(mask, (2, 2, 78, 38))

    assert np.count_nonzero(filtered[:, 12:30]) > 0
    assert np.count_nonzero(filtered[:, 77:80]) == 0


def test_merge_nearby_ocr_blocks_groups_stacked_text_regions():
    blocks = [
        {"text": "Q. 서준쌤이 화내는", "bbox": [79, 435, 601, 501]},
        {"text": "모습을 보신 적\n있으신가요?", "bbox": [144, 517, 536, 679]},
        {"text": "김대수/강산고 영어교사", "bbox": [3, 2081, 553, 2144]},
        {
            "text": "- 글쎄, 워낙 다정한 사람이라\n화내는 모습은 거의 본적 없지만...",
            "bbox": [0, 2154, 633, 2268],
        },
    ]

    merged = merge_nearby_ocr_blocks(blocks)

    assert len(merged) == 2
    assert merged[0]["bbox"] == [79, 435, 601, 679]
    assert merged[0]["text"] == "Q. 서준쌤이 화내는\n모습을 보신 적\n있으신가요?"
    assert merged[0]["_merged_from"] == [0, 1]
    assert merged[1]["bbox"] == [0, 2081, 633, 2268]
    assert merged[1]["_merged_from"] == [2, 3]


def test_reading_order_groups_y_jitter_into_left_to_right_rows_without_mutation():
    blocks = [
        {"text": "right", "bbox": [120, 10, 170, 50]},
        {"text": "next-row", "bbox": [15, 90, 70, 130]},
        {"text": "left", "bbox": [10, 20, 60, 60]},
    ]
    original_bboxes = [list(block["bbox"]) for block in blocks]

    ordered = sort_ocr_blocks_reading_order(blocks)

    assert [block["text"] for block in ordered] == ["left", "right", "next-row"]
    assert blocks[0]["text"] == "right"
    assert [block["bbox"] for block in blocks] == original_bboxes


def test_reading_order_keeps_invalid_bboxes_stable_at_end():
    blocks = [
        {"text": "missing"},
        {"text": "bottom", "bbox": [0, 80, 40, 120]},
        {"text": "invalid", "bbox": [10, 10, 10, 20]},
        {"text": "top", "bbox": [0, 5, 40, 45]},
    ]

    ordered = sort_ocr_blocks_reading_order(blocks)

    assert [block["text"] for block in ordered] == ["top", "bottom", "missing", "invalid"]


def test_merge_same_line_fragments_uses_left_to_right_text_order_with_y_jitter():
    blocks = [
        {"text": "RIGHT", "bbox": [65, 10, 100, 45]},
        {"text": "LEFT", "bbox": [20, 16, 58, 51]},
    ]

    merged = merge_nearby_ocr_blocks(blocks)

    assert len(merged) == 1
    assert merged[0]["text"] == "LEFT\nRIGHT"


def test_merge_nearby_ocr_blocks_does_not_join_separate_touching_bubbles():
    blocks = [
        {"text": "어머~", "bbox": [115, 1001, 164, 1139], "_ocr_index": 2},
        {"text": "시홍아.\n이것도 네가\n만든 거니?", "bbox": [137, 1212, 398, 1398], "_ocr_index": 3},
    ]

    merged = merge_nearby_ocr_blocks(blocks)

    assert len(merged) == 2
    assert merged[0]["text"] == "어머~"
    assert merged[1]["text"] == "시홍아.\n이것도 네가\n만든 거니?"


def test_filter_ocr_blocks_merges_nearby_safe_blocks():
    from app import filter_ocr_blocks

    image = np.full((190, 260, 3), 204, dtype=np.uint8)
    cv2.putText(image, "Q. TEST", (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "LINE TWO", (55, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    blocks = [
        {"text": "Q. TEST", "bbox": [30, 40, 220, 85]},
        {"text": "LINE TWO", "bbox": [50, 95, 210, 140]},
    ]

    filtered, skipped = filter_ocr_blocks(blocks, image, "en")

    assert skipped == 0
    assert len(filtered) == 1
    assert filtered[0]["text"] == "Q. TEST\nLINE TWO"
    assert filtered[0]["_merged_from"] == [0, 1]


def test_scaled_retry_replaces_korean_tall_narrow_ocr_error():
    from ocr.chrome_lens_ocr import _merge_scaled_retry_blocks

    original = [{"text": "~喆喆몽", "bbox": [495, 355, 540, 545]}]
    scaled = [{"text": "움쀼쀼~", "bbox": [418, 430, 608, 473]}]

    refined = _merge_scaled_retry_blocks(original, scaled, (690, 1233), "ko")

    assert refined[0]["text"] == "움쀼쀼~"
    assert refined[0]["bbox"] == [418, 430, 608, 473]
    assert refined[0]["_scaled_retry"] is True


def test_scaled_retry_replaces_hangul_tall_narrow_even_when_language_is_zh():
    from ocr.chrome_lens_ocr import _merge_scaled_retry_blocks

    original = [{"text": "~喆喆몽", "bbox": [493, 349, 542, 551]}]
    scaled = [{"text": "움쀼쀼~", "bbox": [418, 430, 608, 473]}]

    refined = _merge_scaled_retry_blocks(original, scaled, (690, 1233), "zh")

    assert refined[0]["text"] == "움쀼쀼~"
    assert refined[0]["bbox"] == [418, 430, 608, 473]
    assert refined[0]["_scaled_retry"] is True


def test_refine_tall_narrow_ocr_bbox_expands_user_korean_interjection():
    image_path = r"F:\webtruyen-main\webtruyen-main\server\uploads\chapters\1\3\027.webp"
    if not os.path.exists(image_path):
        pytest.skip("local user sample image is not available")
    image = cv2.imread(image_path)

    refined = refine_tall_narrow_ocr_bbox(
        image,
        [117, 1005, 162, 1135],
        source_lang="ko",
    )

    assert refined[0] <= 85
    assert refined[2] >= 200
    assert refined[1] >= 1035
    assert refined[3] <= 1110


def test_refine_tall_narrow_ocr_bbox_uses_hangul_text_when_language_is_zh():
    image_path = r"F:\webtruyen-main\webtruyen-main\server\uploads\chapters\1\3\027.webp"
    if not os.path.exists(image_path):
        pytest.skip("local user sample image is not available")
    image = cv2.imread(image_path)

    refined = refine_tall_narrow_ocr_bbox(
        image,
        [115, 1001, 164, 1139],
        source_lang="zh",
        text="어머~",
    )

    assert refined[0] <= 85
    assert refined[2] >= 200
    assert refined[1] >= 1035
    assert refined[3] <= 1110


def test_assess_erasability_accepts_speech_bubble_text():
    image = np.full((140, 180, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (90, 70), (65, 42), 0, 0, 360, (0, 0, 0), 3)
    cv2.putText(image, "HEY", (52, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    result = assess_erasability(image, [48, 48, 125, 82], text="HEY", source_lang="en")

    assert result["safe"] is True
    assert result["reason"] in {"in_bubble", "uniform_background", "text_like_mask"}
    assert result["score"] >= 0.55


def test_assess_erasability_accepts_flat_background_text():
    image = np.full((90, 150, 3), 245, dtype=np.uint8)
    cv2.putText(image, "TEST", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    result = assess_erasability(image, [10, 20, 120, 65], text="TEST", source_lang="en")

    assert result["safe"] is True
    assert result["reason"] in {"uniform_background", "text_like_mask"}
    assert result["score"] >= 0.55


def test_assess_erasability_accepts_gray_title_card_near_panel_edge():
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 35), (220, 125), (204, 204, 204), -1)
    cv2.putText(image, "Q. TITLE", (32, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5)
    cv2.putText(image, "Q. TITLE", (32, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    result = assess_erasability(image, [25, 58, 215, 92], text="Q. TITLE", source_lang="ko")

    assert result["safe"] is True
    assert result["reason"] == "uniform_background"


def test_assess_erasability_accepts_user_gray_title_card_block():
    image_path = r"C:\Users\dun\Downloads\anhdemo\17774840590457_0296.jpeg"
    if not os.path.exists(image_path):
        pytest.skip("local user sample image is not available")
    image = cv2.imread(image_path)

    result = assess_erasability(
        image,
        [79, 435, 601, 501],
        text="Q. 서준쌤이 화내는",
        source_lang="ko",
    )

    assert result["safe"] is True
    assert result["reason"] == "uniform_background"


def test_erase_text_region_cleans_user_gray_title_card_block():
    image_path = r"C:\Users\dun\Downloads\anhdemo\17774840590457_0296.jpeg"
    if not os.path.exists(image_path):
        pytest.skip("local user sample image is not available")
    image = cv2.imread(image_path)
    bbox = [79, 435, 601, 501]
    x1, y1, x2, y2 = bbox

    erased, _, _ = erase_text_region(image.copy(), bbox, source_lang="ko")
    before_gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(erased[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    assert np.count_nonzero(after_gray > 230) < np.count_nonzero(before_gray > 230) * 0.15
    assert np.count_nonzero(after_gray < 80) < np.count_nonzero(before_gray < 80) * 0.05
    assert after_gray.std() < 0.5


def test_assess_erasability_accepts_user_sfx_inside_white_bubble():
    image_path = r"F:\webtruyen-main\webtruyen-main\server\uploads\chapters\1\3\028.webp"
    if not os.path.exists(image_path):
        pytest.skip("local user sample image is not available")
    image = cv2.imread(image_path)

    result = assess_erasability(
        image,
        [370, 1307, 519, 1346],
        text="크르르",
        source_lang="ko",
    )

    assert result["safe"] is True
    assert result["reason"] == "in_bubble"


def test_decide_skip_render_does_not_filter_sfx_on_complex_artwork():
    should_skip = _decide_skip_render(
        "BOOM",
        {"is_sfx": True, "sfx_type": "short_sfx", "confidence": 1.0},
        {
            "bubble_context": "on_artwork_mixed",
            "uniformity": "complex",
            "intensity_std": 90,
            "edge_score": 120,
        },
        source_lang="en",
    )

    assert should_skip is False


def test_assess_erasability_skips_text_on_complex_artwork():
    image = np.full((120, 180, 3), 180, dtype=np.uint8)
    for x in range(0, 180, 8):
        cv2.line(image, (x, 0), (179 - x // 2, 119), (30 + x % 80, 40, 90), 2)
    for y in range(0, 120, 10):
        cv2.line(image, (0, y), (179, 119 - y // 2), (220, 220 - y % 90, 40), 1)
    cv2.putText(image, "BOOM", (35, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

    result = assess_erasability(image, [25, 35, 145, 82], text="BOOM", source_lang="en")

    assert result["safe"] is False
    assert result["reason"] in {"complex_artwork", "excessive_mask"}
    assert result["score"] < 0.55


def test_assess_erasability_skips_textured_artwork_even_if_bubble_like():
    image = np.full((150, 230, 3), 210, dtype=np.uint8)
    for y in range(5, 150, 8):
        for x in range(5, 230, 8):
            radius = 1 + ((x + y) // 16) % 3
            cv2.circle(image, (x, y), radius, (95, 95, 95), -1)
    cv2.line(image, (0, 20), (229, 130), (45, 45, 45), 2)
    cv2.line(image, (20, 149), (180, 0), (70, 70, 70), 2)
    cv2.putText(image, "TEXT", (55, 88), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

    result = assess_erasability(image, [42, 44, 175, 103], text="TEXT", source_lang="en")

    assert result["safe"] is False
    assert result["reason"] == "complex_artwork"
    assert result["analysis"]["bg_texture_std"] > 55


def test_assess_erasability_accepts_dialogue_in_bubble_with_edge_texture():
    image = np.full((180, 180, 3), 35, dtype=np.uint8)
    cv2.ellipse(image, (92, 92), (70, 62), 0, 0, 360, (248, 248, 248), -1)
    cv2.ellipse(image, (92, 92), (70, 62), 0, 0, 360, (0, 0, 0), 2)
    for idx, text in enumerate(["DIA", "LOG", "UE"]):
        cv2.putText(
            image,
            text,
            (35, 65 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

    result = assess_erasability(
        image,
        [26, 34, 142, 154],
        text="没猜错的话南夏应该是在用天书",
        source_lang="zh",
    )

    assert result["safe"] is True
    assert result["reason"] == "in_bubble"


def test_assess_erasability_does_not_mutate_image():
    image = np.full((90, 150, 3), 245, dtype=np.uint8)
    cv2.putText(image, "TEST", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    original = image.copy()

    assess_erasability(image, [10, 20, 120, 65], text="TEST", source_lang="en")

    assert np.array_equal(image, original)
