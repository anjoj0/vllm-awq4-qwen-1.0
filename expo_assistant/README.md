# 本地编程问答展示页

调用已经运行的 Qwen3.6-27B vLLM 服务，提供多轮问答、流式回答、代码块复制和性能统计。页面不会加载第二份模型，也不会执行生成的代码。

Python 3.9+ 标准库即可运行；项目推理容器使用的 Python 3.12 也可。前端为本地静态文件，无 Node 构建、pip 依赖、CDN、外部字体或云端服务。

## 展示机启动：只需这几步

先按项目原配置启动模型。确认：

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

再复制完整 `expo_assistant/` 目录到展示机，进入该目录运行：

```bash
python3 server.py --upstream http://127.0.0.1:8000 --model Qwen3.6-27B-AWQ4
```

浏览器打开 **http://127.0.0.1:8088**。已有项目的默认服务名称是 `Qwen3.6-27B-AWQ4`；若 `/v1/models` 返回其他名称，修改 `--model`，无需改前端源码。

Windows 上也可用 `python server.py ...` 运行这个页面，调用可访问的 Linux 推理服务。页面能在 Windows 运行，不代表底层 ROCm/vLLM 模型栈获得原生 Windows 支持。

不改现有模型容器、不重编译 vLLM、不下载新权重。首次上机只需确认 API 地址、模型名和输出正常。

## 页面功能

- 多轮编程问答与三个可直接使用的示例问题。
- Markdown、代码块、代码复制、导出当前会话。
- 流式回答、可折叠思考内容、停止生成、新建对话与错误重试。
- 输出长度和思考模式开关，默认关闭思考，减少展示等待。
- 连接状态、输入/输出 token、Prefill/Decode tok/s、首输出延迟和耗时。

对话仅保存在当前页面内存，刷新即清空；导出是主动下载文件，不会上传。长对话仍受模型上下文限制，展会建议每组演示新建对话。

## 为什么默认使用 Responses

本项目现有 `glados.py` 走 `/v1/responses`，用于避开旧 vLLM 0.20 的部分 Chat 流式 reasoning 解析问题。页面默认沿用该接口，并通过同源 Python 代理转发，浏览器无需处理跨域。

若你的后端只有 Chat Completions：

```bash
python3 server.py --upstream http://127.0.0.1:8000 --api-mode chat --model Qwen3.6-27B-AWQ4
```

每种协议需在目标后端单独验收；应用不会在失败后自动重复请求另一协议。若服务需要 API key，在启动进程的环境中设置 `QWEN_API_KEY`；密钥不会发给浏览器或写入聊天导出。

`--upstream` 既可填服务根地址，也可带 `/v1`。模型地址由启动参数决定，页面中的访客不能更改上游地址。

## 速率的口径

生成过程中显示经过时间，最终 usage 和阶段速率在本轮完成后更新。应用不会用字符数、SSE 片段数估算 token。

1. **Prefill tok/s**：优先采用 `Δvllm:request_prefill_kv_computed_tokens_sum / Δvllm:request_prefill_time_seconds_sum`，排除已缓存的输入 token。缺少此指标时，只有 usage 明确报告缓存 token 数才用 `(input_tokens-cached_tokens)/prefill_seconds`。没有可靠分子时显示 `—`。
2. **Decode tok/s**：`(output_tokens-1)/decode_seconds`。vLLM 0.20 的 decode 阶段从首 token 产生后开始计时，首 token 属于 prefill，因此排除；输出≤1或阶段时间无效时显示 `—`。
3. **首输出延迟**：Python 代理发起上游请求到首个正文/思考文本 delta 的时间，包含上游排队与网络影响；它不是纯 GPU prefill 时间。若先产生思考，也计为首输出。
4. **生成耗时**：发起上游生成请求到收到完成事件，不含前后抓取 metrics 的时间。
5. `delivery_tps` 如出现在统计中，仅表示客户端接收窗口内的交付速率估计，不用于冒充服务端 Decode。

vLLM 阶段时长可能包含阶段内抢占等等待，因此也不等于孤立 kernel 的硬件带宽或瞬时峰值。

**为展示测速，请让这一模型服务只接受本页面的单个请求。** 应用内部串行处理，按 model 过滤 Prometheus 样本，等待已完成请求计数更新，并检查 usage 与统计是否一致。但 `/metrics` 本质为模型/engine 聚合，无法阻止其他客户端直连后台。如果检测到并发、计数或 token 不匹配，会隐藏本轮阶段速率。

若 `/metrics` 未开放、指标不完整、全部输入命中缓存或输出过短，相应数字为 `—`，页面仍可正常问答。缺失数据不会填 0 或历史跑分。

参考：[vLLM v0.20 阶段统计实现](https://github.com/vllm-project/vllm/blob/v0.20.0/vllm/v1/metrics/stats.py)、[官方指标](https://docs.vllm.ai/en/v0.20.0/usage/metrics/)。

## 界面预览与常用参数

手边没有 GPU 服务时，可以只预览界面：

```bash
python3 server.py --demo
```

预览有醒目标识，返回固定的模拟文本，**不调用模型、不显示虚构的测速数据**。正式展示去掉 `--demo`。

其他常用参数：

```bash
python3 server.py --port 8088 --max-output-tokens 1024 --timeout 600
```

`--subtitle` 仅修改展示副标题，不改变模型引擎或启用 DFlash。实际推理模式由原 vLLM 启动参数决定。

默认页面只监听本机。确需另一台设备访问，可设置 `--host 0.0.0.0` 并配置实际网络；这不会自动为网页增加登录验证。展会单机显示保持默认即可。

停止按钮关闭上游 HTTP 连接，请求取消的 GPU 回收行为仍由 vLLM 决定。如果后台异常卡住，按原项目的服务恢复流程处理，页面不会直接杀模型进程。

## 开发验证

```bash
python3 -m unittest -v test_server.py
```

测试使用本机假的推理 HTTP 服务验证流式协议、指标口径、错误与取消，不下载模型或运行 GPU。真实雷神机器仍需做一次完整问答与测速验收。

目录：

```text
expo_assistant/
  server.py          Python 标准库代理与阶段统计
  dist/index.html    网页
  dist/app.css       本地样式
  dist/app.js        对话与流式界面
  test_server.py     本地 mock 集成测试
  README.md          本说明
```
