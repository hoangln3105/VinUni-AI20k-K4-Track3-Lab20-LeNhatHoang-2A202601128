# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

Mỗi nhóm trả lời 2 câu:

1. Case nào nên dùng multi-agent? Vì sao?

   Khi task có nhiều bước xử lý khác bản chất nhau (tìm nguồn → phân tích bằng chứng → viết
   câu trả lời có trích dẫn) và cần kiểm soát chất lượng từng bước độc lập — tách vai trò cho
   phép trace "ai làm gì, tốn bao nhiêu", retry/fallback riêng từng bước khi fail, và thêm bước
   phản biện (Critic) kiểm tra citation coverage trước khi trả kết quả. Trong benchmark thực đo
   ở `reports/benchmark_report.md` (chạy với API key thật), bản multi-agent đạt quality
   score 9.6-10.0/10 và citation coverage 100% so với baseline chỉ đạt 7.0/10 và không trích
   dẫn nguồn nào, nhờ có bước Researcher tách riêng và Critic kiểm tra lại. Phù hợp với các
   task cần độ tin cậy cao và có thể audit
   được: báo cáo nghiên cứu, tài liệu kỹ thuật, nội dung cần fact-checking.

2. Case nào không nên dùng multi-agent? Vì sao?

   Khi câu hỏi đơn giản, một lượt là đủ (trả lời FAQ, dịch một câu, tóm tắt ngắn) hoặc khi
   latency là ưu tiên số một (chat trực tiếp cần phản hồi tức thì). Multi-agent gọi LLM nhiều
   lần tuần tự (researcher → analyst → writer → critic) nên tốn nhiều thời gian và chi phí hơn
   hẳn so với 1 lần gọi của baseline, mà với câu hỏi đơn giản thì mức tăng chất lượng đó không
   bù lại được chi phí latency/cost phát sinh — baseline một lượt là lựa chọn hợp lý hơn.
