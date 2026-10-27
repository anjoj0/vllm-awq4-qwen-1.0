"""One-click sample tasks with deterministic local fixtures."""
from pathlib import Path
import base64

ROOT = Path(__file__).resolve().parent / 'examples'
CASES = {
    'project': {
        'title': '项目文件读写', 'mode': 'agent',
        'prompt': '请实际读取 harness-demo/brief.json 和 harness-demo/meeting-notes.txt，生成一份 Markdown 演示准备清单，写入 harness-demo/checklist.md。清单需包含所有 3 项任务的优先级、状态、负责人、目标日期，并统计已完成和待完成数量。写入前检查该文件是否存在；如已存在，先读取后再更新。最后告诉我写入路径和待完成数量。',
        'check': '校验点：3 项任务，已完成 1 项、待完成 2 项；产生 harness-demo/checklist.md，可在工具卡片中打开。',
        'followup': '请读取刚生成的清单，将“联网搜索演示”的负责人改为“现场小组”，保留其他内容并保存。'},
    'multimodal': {
        'title': '图片与文档理解', 'mode': 'chat',
        'prompt': '请结合这张柱状图和周目标文档回答：周一至周五分别完成多少个任务？合计多少？本周是否达标，距离目标还差多少？用简洁表格和一句结论回答。',
        'check': '校验点：每天 32 / 46 / 41 / 59 / 72，合计 250；目标 260，尚差 10。',
        'followup': '如果下周目标仍为 260，平均每天需要完成多少个任务？'},
    'search': {
        'title': '联网搜索与引用', 'mode': 'search',
        'prompt': '请搜索 vLLM 官方的 Automatic Prefix Caching 文档，打开原文，用三个要点说明它对 prefill、decode 的作用和适用场景，并附上官方来源链接。先用英文关键词搜索，优先查找 docs.vllm.ai。',
        'check': '校验点：出现 web_search 与 fetch_page 工具记录；回答包含可点击来源，并区分 prefill 复用与 decode。',
        'followup': '从刚才的官方资料中，找一个适合多轮对话的具体例子。'},
    'weather': {
        'title': '城市天气查询', 'mode': 'search',
        'prompt': '请查询北京当前天气和从今天起 3 天的预报，说明温度、降水概率、数据时间，并给出简短的出行建议和来源链接。',
        'check': '校验点：调用 get_weather，显示匹配城市、当前数据时间与 3 天预报；数值来自实际天气接口。北京仅为示例，可改成你的城市。',
        'followup': '请再查询上海从今天起 3 天的天气，用表格与北京比较温度和降水概率。'}
}


def load_example(agent, session_id, case_id):
    if case_id not in CASES:
        raise ValueError('未知体验案例')
    data = dict(CASES[case_id])
    session = agent.store.load(session_id)
    if case_id == 'project':
        for name in ['brief.json', 'meeting-notes.txt']:
            path = 'harness-demo/' + name
            if not agent.workspace.path(path).exists():
                agent.workspace.write_file(path, (ROOT/name).read_text(), 'new')
    attachments = []
    if case_id == 'multimodal':
        for name in ['weekly-tasks.png', 'weekly-target.txt']:
            attachments.append(agent.store.upload(session_id, name, base64.b64encode((ROOT/name).read_bytes()).decode()))
    session = agent.store.load(session_id)
    session['title'] = data['title']
    agent.store.save(session)
    return {**data, 'id': case_id, 'attachments': attachments}
