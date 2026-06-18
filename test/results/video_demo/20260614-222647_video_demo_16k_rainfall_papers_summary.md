# Context Pressure Test: video_demo_16k_rainfall_papers_summary

## Configuration

| Field | Value |
|---|---:|
| host | `http://127.0.0.1:8001` |
| model | `Qwen3.6-27B-AWQ4` |
| requested prompt tokens | 16000 |
| actual prompt tokens before chat template | 16000 |
| max output tokens | 256 |
| request mode | `nonstream` |
| full prompt tokens | 353625 |

## Result

| Metric | Value |
|---|---:|
| ok | True |
| status | 200 |
| prompt tokens | 16010 |
| completion tokens | 256 |
| TTFT seconds | None |
| wall seconds | 181.47683690800022 |
| prefill tokens/s by TTFT | None |
| stream decode tokens/s | None |
| e2e output tokens/s | 1.4106483469831432 |
| finish reason | `length` |
