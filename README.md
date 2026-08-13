# Manga Translator

Web app Flask để OCR, dịch và render lại chữ trong ảnh manga, manhwa, manhua.

## Chạy nhanh trên Windows

Khuyến nghị dùng Python 3.10 hoặc 3.11. Đừng cài dependency thẳng vào Python global, hãy để project tự tạo môi trường riêng `.venv`.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1
powershell -ExecutionPolicy Bypass -File .\run_app.ps1
```

Mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

Nếu máy có nhiều bản Python, chọn rõ bản muốn dùng:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1 -PythonVersion 3.10
```

Sau khi `.venv` đã tạo xong, lần sau chỉ cần chạy:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_app.ps1
```

## Vì sao dùng `.venv`

Mỗi người có thể đang dùng Python khác nhau, ví dụ 3.9, 3.10, 3.11, 3.12. Một số thư viện xử lý ảnh như `opencv-python`, `numpy`, `pillow` khá nhạy với phiên bản Python.

`.venv` giúp:

- Cài dependency riêng cho project này.
- Không làm hỏng Python global của máy.
- Dễ xóa và cài lại khi dependency lỗi.
- Nhiều project khác nhau không đạp package của nhau.

Nếu môi trường bị lỗi, có thể xóa `.venv` rồi tạo lại:

```powershell
Remove-Item -Recurse -Force .\.venv
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1
```

## Chạy thủ công

Nếu không muốn dùng script:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Trên macOS/Linux:

```bash
python3.10 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python app.py
```

## Chức năng chính

- Upload nhiều ảnh truyện cùng lúc.
- Hỗ trợ ảnh `JPG`, `JPEG`, `PNG`, `WebP`, `BMP`, `TIFF`, `AVIF`.
- OCR bằng Chrome Lens.
- Dừng sau OCR để sửa thủ công block text nếu cần.
- Dịch bằng Gemini, Local LLM hoặc Google Translate.
- Gemini hỗ trợ nhập nhiều API key và tự đổi key khi key bị quota/auth lỗi.
- Xóa text cũ và render text mới vào đúng vùng ảnh.
- Tải từng ảnh hoặc tải toàn bộ bằng file ZIP.

## Cách dùng

1. Chạy app và mở `http://127.0.0.1:5000`.
2. Upload ảnh truyện.
3. Chọn ngôn ngữ nguồn và ngôn ngữ đích.
4. Chọn bộ dịch:
   - `Gemini`: dùng API key Gemini.
   - `Local LLM`: dùng server OpenAI-compatible như LM Studio, Ollama, LocalAI, vLLM.
   - `Google`: dùng `deep-translator`.
5. Bật chỉnh sửa thủ công nếu muốn kiểm tra OCR trước khi dịch.
6. Bấm dịch và chờ progress chạy xong.
7. Tải ảnh kết quả hoặc ZIP.

## Gemini nhiều key

Trong ô Gemini API Keys, nhập nhiều key bằng xuống dòng, dấu phẩy hoặc dấu chấm phẩy:

```text
key_1
key_2
key_3
```

Khi một key bị quota, invalid, unauthorized hoặc permission lỗi, app sẽ thử key tiếp theo trong phiên dịch đó. Nếu tất cả key lỗi, app giữ nguyên text gốc và hiển thị cảnh báo thay vì crash.

Ô `Model Name` trong phần Gemini cho phép nhập model muốn dùng. Giá trị mặc định là `gemini-3.1-flash-lite` và được lưu trong trình duyệt cho lần sử dụng tiếp theo.

## Local LLM

Local LLM cần server tương thích OpenAI `/v1/chat/completions`.

Ví dụ:

- LM Studio: `http://localhost:1234`
- Ollama OpenAI-compatible: `http://localhost:11434`
- LocalAI/vLLM: tùy cấu hình server của bạn

Điền thêm tên model đúng với server, ví dụ:

```text
qwen2.5
llama3.2
mistral
```

## Biến môi trường

Có thể chỉnh khi chạy app:

```powershell
$env:HOST="127.0.0.1"
$env:PORT="5000"
$env:FLASK_DEBUG="1"
$env:MAX_UPLOAD_MB="50"
$env:SESSION_TTL_SECONDS="21600"
$env:SECRET_KEY="change-me"
powershell -ExecutionPolicy Bypass -File .\run_app.ps1
```

Mặc định:

- `HOST=127.0.0.1`
- `PORT=5000`
- `FLASK_DEBUG=0`
- `MAX_UPLOAD_MB=50`
- `SESSION_TTL_SECONDS=21600`

## Docker

Docker là cách tách biệt mạnh nhất vì không dùng Python của máy host.

Trên Windows có thể build bằng script:

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

Build rồi chạy luôn:

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1 -Run
```

Muốn đổi Python trong container:

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1 -PythonVersion 3.10
```

Lệnh Docker thủ công:

```bash
docker build -t manga-translator .
docker run -p 7860:7860 manga-translator
```

Mở:

```text
http://127.0.0.1:7860
```

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest translator\test_translator.py -q
```

Nếu chưa có `.venv`, chạy `setup_venv.ps1` trước.

## Dọn file tạm

Các thư mục/file runtime không nên commit:

- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- `temp_sessions/`
- `debug_outputs/`
- `server_*.log`

## Ghi chú

- Nên dùng Python 3.10 hoặc 3.11 cho ổn định dependency.
- Nếu dùng Python 3.12/3.13 và cài lỗi, hãy tạo lại `.venv` bằng Python 3.10 hoặc 3.11.
- Khi đổi dependency trong `requirements.txt`, chạy lại `setup_venv.ps1`.
