# Design Template

## Problem

Xây dựng research assistant nhận một câu hỏi nghiên cứu dài, tự tìm nguồn, phân tích nội dung tìm được, và viết câu trả lời cuối cùng có trích dẫn — đồng thời so sánh chất lượng/latency/cost với một single-agent baseline làm toàn bộ trong một lần gọi LLM.

## Why multi-agent?

Single-agent baseline gộp tìm kiếm, phân tích và viết vào một prompt duy nhất nên khó kiểm soát chất lượng từng bước: không có cấu trúc "nguồn nào được dùng", không có bước phản biện độc lập, và không thể retry riêng một giai đoạn khi nó fail. Multi-agent tách rõ trách nhiệm (Researcher tìm nguồn, Analyst đánh giá bằng chứng, Writer tổng hợp, Critic kiểm tra trích dẫn) nên mỗi bước có thể được trace, retry, và benchmark độc lập — đổi lại latency cao hơn do nhiều lượt gọi LLM tuần tự.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Quyết định bước tiếp theo (researcher/analyst/writer/critic/done), enforce max_iterations | `ResearchState` hiện tại | route tiếp theo ghi vào `route_history` | Loop vô hạn nếu thiếu guardrail → chặn bằng `max_iterations` và `max_errors` |
| Researcher | Tìm nguồn (SearchClient) và tóm tắt thành research notes có trích dẫn [n] | `request.query` | `state.sources`, `state.research_notes` | Search/LLM lỗi → retry 2 lần rồi fallback text, ghi vào `state.errors` |
| Analyst | Rút key claims, so sánh quan điểm, gắn cờ bằng chứng yếu | `state.research_notes` | `state.analysis_notes` | Thiếu research_notes → `AgentExecutionError`; LLM lỗi → retry rồi fallback |
| Writer | Tổng hợp câu trả lời cuối cùng, trích dẫn nguồn [n] | `state.analysis_notes`, `state.sources` | `state.final_answer` | Thiếu analysis_notes → `AgentExecutionError`; LLM lỗi → retry rồi fallback |
| Critic | Kiểm tra citation coverage và độ dài câu trả lời (không gọi LLM) | `state.final_answer`, `state.sources` | `state.critic_notes` | Thiếu final_answer → `AgentExecutionError` |

## Shared state

`ResearchState` (xem [state.py](../src/multi_agent_research_lab/core/state.py)):

- `request`: câu hỏi gốc + số nguồn tối đa + audience — bất biến trong suốt workflow.
- `iteration` / `route_history`: đếm số bước supervisor đã điều phối, dùng để enforce `max_iterations` và để debug đường đi.
- `sources`, `research_notes`, `analysis_notes`, `critic_notes`, `final_answer`: kết quả tăng dần của từng agent — supervisor dùng chính các field này (còn `None` hay không) để quyết định route tiếp theo, nên state đóng vai trò "single source of truth" thay vì mỗi agent giữ state riêng.
- `agent_results`: lịch sử output + metadata (token, cost) của từng agent, dùng để tính benchmark cost.
- `trace`: span thời gian từng bước, dùng để giải thích "ai làm gì, tốn bao nhiêu".
- `errors`: lỗi đã được retry/fallback, dùng để quyết định dừng sớm và để tính `failure_rate`.

## Routing policy

```text
START -> supervisor
supervisor --researcher--> researcher -> supervisor
supervisor --analyst-----> analyst    -> supervisor
supervisor --writer------> writer     -> supervisor
supervisor --critic------> critic     -> supervisor
supervisor --done--------> END
```

Supervisor (`SupervisorAgent.decide`, xem [supervisor.py](../src/multi_agent_research_lab/agents/supervisor.py)) chọn route theo thứ tự ưu tiên:

1. `iteration >= max_iterations` → `done` (circuit breaker).
2. `len(errors) >= max_errors (3)` → `done` (best-effort, dừng sớm khi lỗi lặp lại).
3. `research_notes is None` → `researcher`
4. `analysis_notes is None` → `analyst`
5. `final_answer is None` → `writer`
6. `critic_notes is None` → `critic`
7. ngược lại → `done`

## Guardrails

- Max iterations: `settings.max_iterations` (default 6, cấu hình qua `.env`/`configs/lab_default.yaml`), enforce trong `SupervisorAgent.decide`.
- Timeout: `settings.timeout_seconds` cấu hình sẵn cho provider thật (OpenAI client hỗ trợ timeout qua SDK khi cắm key thật).
- Retry: mỗi worker node được wrap bằng `tenacity.Retrying` (2 lần) trong [workflow.py](../src/multi_agent_research_lab/graph/workflow.py); `LLMClient.complete` retry riêng 3 lần với backoff.
- Fallback: khi worker fail hết số lần retry, node gán fallback text vào field tương ứng (`research_notes`/`analysis_notes`/`final_answer`/`critic_notes`) thay vì crash graph, và ghi lỗi vào `state.errors`.
- Validation: `ResearchQuery` (Pydantic) validate độ dài query và `max_sources` tại CLI; `AnalystAgent`/`WriterAgent`/`CriticAgent` raise `AgentExecutionError` nếu gọi thiếu tiền điều kiện (dùng khi test/gọi trực tiếp ngoài graph).

## Benchmark plan

- Query set: lấy từ `configs/lab_default.yaml` (`benchmark.queries`), gồm 3 câu hỏi nghiên cứu đại diện.
- Metric: latency (wall-clock), estimated cost (tổng `cost_usd` từ `agent_results`), quality (heuristic 0-10 dựa trên độ dài + citation coverage — thay bằng điểm peer review thật khi có), citation coverage (tỉ lệ `[n]` xuất hiện trong final answer / tổng số nguồn), failure rate (có lỗi được recover hay không).
- Expected outcome: multi-agent có citation coverage và quality cao hơn baseline (do tách bước Analyst/Critic), đổi lại latency cao hơn vì nhiều lượt gọi LLM tuần tự — khớp với kết quả thực đo trong `reports/benchmark_report.md`.
