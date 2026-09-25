# L3A Architecture Record

Tài liệu này mô tả các quyết định có thể kiểm chứng của workflow. Trace chỉ ghi sự kiện,
decision code và evidence reference; không ghi prompt, dữ liệu bí mật hay chain-of-thought.

## 1. System overview

```text
inputs/<case_id>.json
          |
          v
  Coordinator / Router
          |
          +----------+----------------+----------------+
          v          v                v                v
   Order/Item     Payment          Shipment         Policy
      Agent        Agent             Agent           Agent
          +----------+----------------+----------------+
                             |
                      scoped MCP evidence
                             |
                             v
                       Verifier Agent
                             |
                 schema-valid output + trace
```

Coordinator chỉ dùng claim để chọn nhóm bằng chứng cần thu thập. Customer message không
được xem là ground truth. Kết luận chỉ được chấp nhận khi dữ liệu MCP corroborate claim;
policy MCP quyết định trạng thái, hành động và số tiền hoàn.

Workflow được triển khai bằng Python async state machine. Các specialist độc lập chạy đồng
thời trong phạm vi một case. Mỗi case hoàn tất xác minh trước khi CLI chuyển sang case tiếp
theo, do đó evidence không thể bị dùng chéo case.

## 2. Agent ownership và tool permissions

| Actor | Input | Trách nhiệm | Tool được phép | Output/handoff |
| --- | --- | --- | --- | --- |
| Coordinator | Case envelope | Validate routing fields, lập tool plan, fan-out/fan-in | Không gọi MCP | Agent tasks; handoff sang verifier |
| Order/Item Agent | `case_id`, `order_id` | Trạng thái order, item total, item/seller identity | `get_order`, `get_order_items`, `get_sellers` | Evidence report sang coordinator |
| Payment Agent | `case_id`, `order_id` | Payment capture, mismatch, duplicate, refund lifecycle | `get_order_payments`, `get_payment_timeline`, `get_refund_timeline` | Evidence report sang coordinator |
| Shipment Agent | `case_id`, `order_id` | Delivery deadline, actual delivery và responsible actor | `get_shipment_summary` | Evidence report sang coordinator |
| Policy Agent | `case_id`, `policy_version` | Lấy rule, status, action, refund và party type | `get_policy` | Policy decision sang coordinator |
| Verifier Agent | Candidate output + collected refs | Kiểm tra provenance, consistency và public schema | Không gọi MCP | Validated output hoặc reject |

Tool phải vừa nằm trong allowlist của actor, vừa xuất hiện trong MCP discovery. Coordinator
không cấp quyền tổng quát cho specialist.

## 3. A2A protocol

Logical task envelope gồm:

```text
case_id, sender, recipient, intent, tool requests, scoped arguments
```

`case_id` là correlation key bắt buộc trong mọi MCP call và trace event. Handoff chỉ đi theo
đồ thị hữu hạn `coordinator -> specialist -> coordinator -> verifier -> coordinator`; specialist
không handoff trực tiếp cho nhau, vì vậy không có vòng lặp A2A.

Observable lifecycle cho mỗi case:

1. CLI emit `case_received`.
2. Coordinator emit `task_assigned` cho từng specialist được route.
3. Specialist emit `tool_result_consumed` cho mỗi envelope MCP hợp lệ.
4. Specialist emit `handoff` với `EVIDENCE_READY` hoặc `PARTIAL_EVIDENCE`.
5. Policy Agent emit `policy_decided`.
6. Coordinator emit `handoff` sang Verifier.
7. Verifier emit `verification_completed`; CLI emit `case_finalized`.

Không có nội dung suy luận riêng trong trace. `attributes` chỉ chứa count, trạng thái và mã
quyết định quan sát được.

## 4. Routing và evidence selection

| Nhóm claim | Evidence tối thiểu được route |
| --- | --- |
| Canceled/unavailable paid | order, payment rows, policy; unavailable thêm items |
| Late delivery | order, items, shipment summary, policy |
| Split/mismatch/duplicate payment | order, items, payment timeline, policy |
| Refund pending/failed | order, payment rows, refund timeline, policy |
| Unsupported claim | order, items, payment timeline, shipment summary, policy |

Dataset có thể chứa row nhiễu. Quy tắc chọn dữ liệu:

- item trùng ID: chọn row có shipping limit gần purchase timestamp nhất;
- payment event: ưu tiên event trong cửa sổ hai ngày quanh purchase timestamp;
- late-delivery actor chỉ hợp lệ khi actual delivery trễ estimate và event timestamp khớp
  actual delivery;
- split/duplicate/mismatch được đối chiếu giữa captured total và item + freight total;
- customer claim bất đồng với evidence thì evidence MCP thắng.

## 5. Evidence lifecycle

1. Gateway discovery cache danh sách tool được server quảng bá.
2. Mọi call luôn truyền đúng `case_id` và chỉ dùng argument từ case hiện tại.
3. Gateway validate response bằng `mcp-evidence-response-v1.schema.json`.
4. Specialist giữ nguyên `evidence_ref`; không sửa, hash lại hay tự tạo ref.
5. Chỉ response hợp lệ mới được emit `tool_result_consumed`.
6. Verifier yêu cầu mọi claim ref là tập con của output refs, và mọi output ref là tập con
   của refs đã thu trong case hiện tại.
7. State evidence chỉ sống trong một lần gọi `solve_case`; không có cache evidence liên case.

## 6. Failure policy

| Failure | Retry | Fallback | Trace event/code |
| --- | --- | --- | --- |
| Network/MCP transient error | Tối đa 2 lần, backoff 250 ms | Specialist trả partial report | `handoff/PARTIAL_EVIDENCE` |
| Tool không có trong discovery | Không | Không gọi tool; ghi partial report | `handoff/PARTIAL_EVIDENCE` |
| Tool ngoài allowlist actor | Không | Reject request trong specialist | `handoff/PARTIAL_EVIDENCE` |
| Not found/empty authoritative data | Tối đa theo cùng bounded retry | `insufficient_evidence` | `handoff/PARTIAL_EVIDENCE` |
| Claim và MCP xung đột | Không | Chọn MCP, ghi `data_conflicts` | `verification_completed/OUTPUT_VALID` |
| Invalid output/foreign evidence ref | Không | Verifier raise và không finalize case | Không emit completion |

Retry chỉ lặp một call idempotent với cùng `case_id` và arguments. Missing evidence không được
chuyển thành dữ liệu phỏng đoán.

## 7. Verification invariants

Trước finalize, Verifier kiểm tra:

- output pass nguyên bản `l3a-output-v2.schema.json` với `additionalProperties: false`;
- output `case_id` bằng case đang xử lý;
- mọi evidence ref thuộc tập đã thu cho đúng case và đã xuất hiện trong
  `tool_result_consumed` của chính case đó;
- claim refs là tập con của top-level evidence refs;
- tổng `refund_lines.amount_brl` bằng `recommended_refund_brl`;
- `no_action` không có refund dương;
- resolution actions không trùng;
- seller/item/order IDs chỉ lấy từ evidence hoặc input scope;
- confidence nằm trong schema bounds;
- policy action, case status, refund và responsible party type nhất quán.

Artifact validator cũng yêu cầu đủ `case_received`, `task_assigned`, `handoff`,
`verification_completed`, `case_finalized` và kiểm tra receive xảy ra trước finalize.

Business resolution lấy từ policy envelope do `get_policy` trả về. File
`contracts/scoring/scoring-policy-v2.json` chỉ là scoring contract: workflow đọc danh sách
event bắt buộc từ file này, không diễn giải scoring weights thành luật hoàn tiền.

Verifier khóa mapping issue/status/party/action, ví dụ:

- `late_delivery_seller` → `action_required` → `seller` → `refund_freight`;
- `late_delivery_logistics` → `action_required` → `logistics_provider` → `refund_freight`;
- payment/refund failures → `payment_provider`;
- valid split hoặc unsupported claim → `no_action` → `customer`.

Confidence bắt đầu từ 0.96 cho kết luận được corroborate, 0.92 cho unsupported claim và thấp
hơn cho insufficient evidence. Điểm sau đó giảm theo tỷ lệ tool thành công, MCP warnings và
số conflict đã resolve (0.04 mỗi conflict, 0.02 mỗi warning, tổng penalty tối đa 0.30).
Verifier cấm confidence lớn hơn 0.95 khi còn data conflict và lớn hơn 0.55 khi evidence thiếu.

Public schemas trong `contracts/schemas/` là read-only contract. Workflow không thêm field và
không thay đổi ý nghĩa schema.

## 8. LLM reasoning layer (Qwen 3 8B via OpenRouter)

Workflow tích hợp Qwen 3 8B (< 10B parameters) thông qua OpenRouter API, đáp ứng yêu cầu
bắt buộc sử dụng model dưới 10B.

### Architecture

```text
.env (OPENROUTER_API_KEY, OPENROUTER_MODEL, ...)
         |
         v
   load_llm_client()  →  LLMClient (httpx2-based async wrapper)
         |
         +---- analyze_evidence()      ← mỗi specialist gọi sau khi thu evidence
         +---- synthesize_assessment() ← coordinator gọi trước khi build output
```

### Thiết kế

- **Advisory layer**: LLM phân tích evidence và tạo structured JSON summary. Kết quả chỉ
  được log vào trace (`llm_assisted: true`), không thay thế decision rules deterministic.
- **Graceful degradation**: Nếu `OPENROUTER_API_KEY` không có hoặc LLM call thất bại,
  workflow tiếp tục bình thường với logic deterministic.
- **Role-specific prompts**: Mỗi specialist (order, payment, shipment, policy) và coordinator
  có system prompt riêng, yêu cầu output JSON cấu trúc.
- **Qwen 3 thinking**: Client tự động strip `<think>…</think>` tags khi model bật thinking.
- **Retry**: 2 lần với exponential backoff trên lỗi HTTP tạm thời.

### Modules

| File | Trách nhiệm |
| --- | --- |
| `llm_client.py` | Async HTTP client cho OpenRouter chat completions |
| `llm_reasoning.py` | System prompts và hàm reasoning cho từng specialist role |
| `config.py` | `load_llm_client()` đọc cấu hình từ `.env` |

### Trace observability

Khi LLM được kích hoạt, trace attributes ghi nhận:
- `llm_assisted: true` trên handoff event của specialist khi LLM analysis thành công
- `llm_model: "qwen/qwen3-8b"` trên coordinator → verifier handoff

API key không bao giờ xuất hiện trong trace, output hay submission.

## 9. Reproducibility

- Runtime: Python 3.11+; môi trường hiện tại dùng Python 3.12.
- Dependency ranges được khóa trong `pyproject.toml`; cài bằng `pip install -e ".[dev]"`.
- LLM: Qwen 3 8B qua OpenRouter; cấu hình trong `.env` (xem mục 8).
- Decision rules deterministic; LLM chỉ là advisory layer, không ảnh hưởng output cuối cùng.
- Concurrency: tối đa số specialist được route trong một case; case được CLI xử lý tuần tự.
- MCP retries: 2; backoff: 0.25 giây. LLM retries: 2; backoff: 1 giây.
- Chạy: `day09 run`, `day09 validate`, `day09 package --output dist/submission.zip`.
- API key chỉ đọc từ `.env`, không ghi vào trace, output, manifest hay tài liệu này.

