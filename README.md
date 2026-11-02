# DataMate task 2 — baseline

Trả lời câu hỏi tiếng Việt thường ngày trên sổ sách thật của tiệm nhỏ
(Zalo AI Challenge 2026, track 2). Mỗi câu hỏi đi kèm `as_of_date` và dữ liệu của
một shop ở dạng CSV / XLSX / SQLite; đáp án phải là **giá trị có type**
(`number`, `percent`, `string`, `date`, `boolean`, `list`, `table`,
`unanswerable`), chấm tuyệt đối: đúng +1, sai −1, bỏ trống 0.

## Cách tiếp cận

Model **không tự nói ra con số**. Với mỗi câu, nó nhận rulebook, README của shop
và preview thô của mọi bảng, rồi **viết một đoạn pandas**; đoạn code chạy trên
file thật của shop, và kết quả nó tính được mới là đáp án.

```
câu hỏi + as_of_date
  ├─ rulebook.md của split        (nguyên văn, không tóm tắt)
  ├─ README.md của shop           (shop tự mô tả cột)
  └─ preview THÔ mỗi bảng/sheet   (6 dòng đầu, chưa áp header)
        │
        ▼  LLM
  đoạn ```python ... ANSWER = {"type": ..., "answer": ...}
        │
        ▼  subprocess có timeout
  chạy trên file thật
        ├─ traceback / ANSWER sai format → gửi lỗi lại cho model, sửa (tối đa 2 lần)
        └─ hợp lệ → record + submission
             hết lượt sửa → {"type": "abstain"}  (0 điểm, thay vì đoán và −1)
```

Vì sao preview là **thô** (`header=None`): đó là cách duy nhất để model thấy
"dòng 0 là tiêu đề báo cáo, header thật ở dòng 3", header 2 dòng, `55k`,
`45689` (serial ngày Excel) — rồi tự chọn `header_row` và tự làm sạch. Nếu loader
tự đoán header thì nó phải đoán đúng cho cả 44 shop, và đoán sai thì model không
có cách nào biết.

Vì sao pandas chứ không SQL: 43/44 shop không phải database. Phần nặng của bài là
đọc sổ bẩn (title row, fill-down, quy đổi đơn vị, map trạng thái), không phải
aggregation. Shop có `pos.sqlite` vẫn dùng được `sql(...)` trong sandbox.

## Cài đặt

```bash
cd /GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ZaloAI_Opencubee2/workspace/enn/task2
uv sync
cp .env.example .env     # sửa DATAMATE_BASE_URL / DATAMATE_MODEL
```

## Chạy

```bash
# thử 20 câu một shop trước
uv run python inference.py ../../../dataset/track_2/train/train \
    --business biz_101 --limit 20 --out runs/smoke.jsonl
uv run python evaluate.py ../../../dataset/track_2/train/train runs/smoke.jsonl --only-answered

# cả split train (có đáp án để đo), rồi public test (để nộp)
uv run python inference.py ../../../dataset/track_2/train/train --out runs/train.jsonl --workers 8
uv run python evaluate.py  ../../../dataset/track_2/train/train runs/train.jsonl --mistakes 20

uv run python inference.py ../../../dataset/track_2/public_test/public_test \
    --out runs/public.jsonl --workers 8
# nộp runs/public.submission.jsonl
```

`inference.py` **resume được**: câu đã có trong `--out` thì bỏ qua, nên chạy lại
sau khi bị ngắt không tốn token lần hai. Xoá một dòng record để bắt tính lại câu đó.

Xem đúng những gì model nhìn thấy cho một shop:

```bash
uv run python -c "from utils.agent import Agent; \
print(Agent('../../../dataset/track_2/train/train').profile('biz_105'))"
```

## File ra

| File | Nội dung |
|---|---|
| `--out` (records) | mỗi dòng một câu: câu hỏi, `as_of_date`, **đoạn code model sinh ra**, **reply cuối cùng của model**, các lần chạy lỗi kèm traceback, và đáp án đã nộp. Đây là file để đọc khi một câu bị −1 |
| `<out>.submission.jsonl` | đúng 3 field `id`, `type`, `answer` để nộp. Ghi lại toàn bộ từ records ở cuối mỗi lần chạy nên không bao giờ lệch với records |
| `.cache/<split>/<biz>/*.pkl` | bảng đã parse (XLSX 14 MB mất ~30 s, mọi câu của shop dùng lại) |
| `.cache/<split>/snippets/<qid>_<lần>.py` | script đã chạy thật, chạy tay lại được bằng `python <file>` |

## Cấu trúc

```
inference.py        chạy một split → records.jsonl (resume) + submission.jsonl
evaluate.py         chấm records/submission → Penalized Accuracy, tách theo type và theo shop
utils/tables.py     đọc CSV/XLSX/SQLite của một shop thành grid thô, cache pickle, dựng preview
utils/sandbox.py    chạy snippet trong subprocess có timeout; to_jsonable chặn đáp án mất kiểu
utils/agent.py      prompt (rulebook + README + preview) + vòng sửa lỗi + fallback abstain
utils/scoring.py    cổng format (§9) + so khớp như grader + Penalized Accuracy
utils/jsonl.py      đọc/ghi JSONL append, flush từng dòng để resume được
tests/test_scoring.py  self-check cho format gate, so khớp, to_jsonable, extract_code
```

Hai nơi quyết định điểm, đọc trước khi sửa gì khác:

- `utils/scoring.py` — `validate()` chặn payload sai format **trước khi** nộp
  (tiền viết dạng chuỗi, percent viết dạng tỉ lệ, string bọc trong list). Sai
  format là −1, nên payload sai format bị hạ thành abstain (0).
- `utils/agent.py` — `SYSTEM_PROMPT` chứa nguyên rulebook cộng danh sách bẫy đọc
  file. Mọi thay đổi về cách hiểu luật nên vào đây hoặc vào snippet, không vào loader.

## Kiểm chứng

- `uv run python tests/test_scoring.py` — 9 self-check: format gate,
  so khớp number/percent/string/list/table, alias theo vị trí, unanswerable vs
  abstain, Penalized Accuracy, `to_jsonable` từ chối Timestamp/NaN/Series,
  `extract_code` lấy block cuối. **Đã chạy, pass.**
- End-to-end không qua LLM: snippet viết tay cho hai câu train của `biz_101`
  (`q_2a3019504577`, `q_8a587b98bf88`) qua `tables → sandbox → validate → is_correct`
  ra đúng `3530400` và `8697200` như `answers.jsonl`. **Đã chạy, khớp.** (Chạy ở
  layout cũ, trước khi gom code vào `utils/`; logic không đổi, nhưng chưa chạy lại
  sau refactor.)
- **Chưa chạy**: vòng gọi LLM thật. `localhost:4308` lúc dựng baseline bị reset
  kết nối (port do container `opencubee2` map ra, server bên trong chưa chạy), nên
  `Agent._complete` chưa được gọi lần nào với endpoint thật.

## Giới hạn đã biết

- **Chưa có policy abstain theo độ tin cậy.** Chỉ abstain khi lỗi kỹ thuật. Với
  PA thì đáp án có xác suất đúng dưới 50% nên bỏ trống — cần đo trên train trước
  khi thêm.
- **Mỗi câu một lần sinh code**, không self-consistency, không vote.
- **Chưa chuẩn hoá về một sổ đơn logic chung** (orders / order_lines / payments /
  imports / lots / stock). Đó là bước tăng điểm tiếp theo: khi có schema sạch thì
  prompt ngắn hơn, và tầng trả lời chuyển sang SQL được.
- **`unanswerable` phụ thuộc hoàn toàn vào model.** Rulebook §7 phân biệt
  "không có dữ liệu" với "bằng 0"; baseline không kiểm tra độc lập khoảng ngày có
  dữ liệu của shop.
- **Preview 6 dòng** có thể không đủ cho sheet mà header thật nằm sâu hơn; tăng
  `DATAMATE_PREVIEW_ROWS` nếu thấy model chọn sai `header_row`.
