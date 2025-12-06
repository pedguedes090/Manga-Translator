---
title: Manga Translator
emoji: 📚
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
license: mit
---
# Manga Translator 📚

Dịch tự động speech bubbles trong manga/manhwa/manhua!

## ✨ Features

### Core
- 🔍 **YOLO-based bubble detection** - Phát hiện speech bubble tự động
- 📝 **Multiple OCR engines** - Manga-OCR, Chrome Lens (batch support)
- 🌐 **Multiple translators** - Gemini, Copilot API, NLLB, Opus-MT

### Translation
- 🧠 **Context Memory** - Sử dụng context từ tất cả ảnh để dịch chính xác hơn
- 🎯 **Multi-page batch translation** - Dịch 10 pages/API call tiết kiệm quota
- 🎨 **Translation styles** - Default, Casual, Formal, Keep Honorifics, Web Novel...

### UI/UX
- 📊 **Real-time progress** - Progress bar hiển thị tiến độ theo từng phase
- 📦 **Download ZIP** - Tải tất cả ảnh đã dịch dưới dạng ZIP
- 🔤 **Auto font sizing** - Tự động điều chỉnh cỡ chữ theo bubble
- 📏 **24+ fonts** - Yuki fonts, AnimeAce, và nhiều font khác

## 🚀 Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

Mở http://localhost:5000

## 📋 Workflow

1. Upload manga/manhwa images
2. Chọn ngôn ngữ gốc (Japanese/Chinese/Korean/English)
3. Chọn ngôn ngữ đích (Vietnamese, English, ...)
4. Chọn translator (Gemini/Copilot) và OCR engine
5. Check "Context Memory" để dịch chính xác hơn
6. Click **Translate**!
7. Xem progress bar real-time
8. Download từng ảnh hoặc **Download ZIP**

## 🌍 Supported Languages

| Source | Target |
|--------|--------|
| Japanese (Manga) | Vietnamese |
| Chinese (Manhua) | English |
| Korean (Manhwa) | Chinese |
| English (Comic) | Korean, Thai, Indonesian, French, German, Spanish, Russian |

## 📡 API Keys

- **Gemini**: Nhập API key từ [ai.google.dev](https://ai.google.dev)
- **Copilot**: Chạy server [copilot-api](https://github.com/copilot-api) local

## 🔧 Tech Stack

- Flask + Flask-SocketIO (real-time WebSocket)
- YOLOv8 (bubble detection)
- Manga-OCR / Chrome-Lens (OCR)
- Gemini / Copilot API (translation)
- PIL (text rendering)
