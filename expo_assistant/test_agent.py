"""Agent boundaries and multi-step integration; no GPU/network required."""
import base64
from contextlib import contextmanager
import io
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

from test_server import running_app, post, decode_events, Plan, chat_events, proxy
from workspace import Workspace, Store, sha, parse_attachment
from web_tools import public_address, fetch_page, get_weather


def tool_events(name, args, finish='tool_calls', done=True):
    raw = json.dumps(args)
    events = [('', {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_1',
                'type': 'function', 'function': {'name': name[:3], 'arguments': raw[:8]}}]}}]}),
              ('', {'choices': [{'delta': {'tool_calls': [{'index': 0,
                'function': {'name': name[3:], 'arguments': raw[8:]}}]}, 'finish_reason': finish}]}),
              ('', {'choices': [], 'usage': {'prompt_tokens': 120, 'completion_tokens': 11,
                                            'prompt_tokens_details': {'cached_tokens': 40}}})]
    return events + ([('', '[DONE]')] if done else [])


def answer_events(content, finish='stop'):
    events = chat_events()
    events[1][1]['choices'][0]['delta']['content'] = content
    events[2][1]['choices'][0]['finish_reason'] = finish
    return events


@contextmanager
def harness():
    with tempfile.TemporaryDirectory() as d, running_app(mode='chat') as (app, fake):
        root = Path(d)/'project'
        root.mkdir()
        app.app.config.workspace = str(root)
        app.app.config.data_dir = str(Path(d)/'data')
        app.app.agent = proxy.Agent(app.app, proxy.make_stats, proxy.iter_sse)
        session = app.app.agent.store.create()
        yield app, fake, root, session


def body(session, mode='agent'):
    return {'request_id': 'test-agent', 'session_id': session['id'], 'text': '按要求完成任务',
            'mode': mode, 'max_tokens': 128, 'thinking': False, 'attachments': []}


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/'project'
        self.root.mkdir()
        self.ws = Workspace(self.root, Path(self.temp.name)/'data')

    def tearDown(self):
        self.temp.cleanup()

    def test_paths_secrets_symlinks_and_binary_are_rejected(self):
        for name in ['../escape', '/etc/passwd', '.git/config', '.env', '.env.production', 'priv.key']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.ws.read_file(name)
        (self.root/'link').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): self.ws.write_file('link/x', 'bad', 'new')
        (self.root/'binary.dat').write_bytes(b'a\0b')
        with self.assertRaises(ValueError): self.ws.read_file('binary.dat')

    def test_stale_hash_prevents_overwrite_and_backup_matches(self):
        p = self.root/'source.txt'
        p.write_text('hello\n')
        read = self.ws.read_file('source.txt')
        p.write_text('changed externally\n')
        with self.assertRaises(ValueError): self.ws.write_file('source.txt', 'lost', read['sha256'])
        current = self.ws.read_file('source.txt')
        result = self.ws.edit_file('source.txt', 'externally', 'safely', current['sha256'])
        self.assertEqual(p.read_text(), 'changed safely\n')
        self.assertIn('-changed externally', result['diff'])
        self.assertEqual((self.ws.data_dir/'changes'/result['change_id']/'before').read_text(), 'changed externally\n')
        with self.assertRaises(ValueError): self.ws.write_file('source.txt', 'bad', 'new')

    def test_search_excludes_blocked_files_and_returns_real_lines(self):
        (self.root/'ok.txt').write_text('one\nneedle\nthree')
        (self.root/'.env').write_text('needle secret')
        (self.root/'secret').symlink_to('/etc/passwd')
        result = self.ws.search_files('needle')
        self.assertEqual(result['matches'], [{'path': 'ok.txt', 'line': 2, 'text': 'needle'}])

    def test_attachments_office_images_and_session_scope(self):
        from PIL import Image
        store = Store(Path(self.temp.name)/'data')
        a, b = store.create(), store.create()
        item = store.upload(a['id'], '../../notes.md', base64.b64encode('你好\n附件测试'.encode()).decode())
        self.assertEqual(item['name'], 'notes.md')
        self.assertIn('附件测试', store.read_attachment(store.load(a['id']), item['id'])['text'])
        with self.assertRaises(ValueError): store.attachment(b, item['id'])
        with self.assertRaises(ValueError): store.upload(a['id'], 'bad.png', 'invalid base64')
        image = io.BytesIO(); Image.new('RGB', (2000, 1000), 'red').save(image, 'PNG')
        info = store.upload(a['id'], 'image.png', base64.b64encode(image.getvalue()).decode())
        self.assertEqual(info['dimensions'], [1280, 640])
        self.assertIn('image_url', store.user_content(store.load(a['id']), '看图', [info['id']])[2]['type'])
        doc = io.BytesIO()
        with zipfile.ZipFile(doc, 'w') as z:
            z.writestr('word/document.xml', '<w:document xmlns:w="urn:w"><w:p><w:r><w:t>Office content</w:t></w:r></w:p></w:document>')
        self.assertIn('Office content', parse_attachment('test.docx', doc.getvalue())[1])
        with self.assertRaises(ValueError): parse_attachment('bad.pdf', b'garbage')

    def test_attachment_search_locates_pages_and_keeps_session_boundary(self):
        store = Store(Path(self.temp.name)/'data')
        owner, other = store.create(), store.create()
        text = '--- 第 1 页 ---\nIntroduction\n--- 第 2 页 ---\nMethod [ablation]\n--- 第 3 页 ---\nResults'
        item = store.upload(owner['id'], 'paper.txt', base64.b64encode(text.encode()).decode())
        owner = store.load(owner['id'])
        found = store.search_attachment(owner, item['id'], '[ABLATION]')['matches'][0]
        self.assertEqual(found['page'], 2)
        self.assertEqual(found['offset'], text.index('[ablation]'))
        self.assertEqual(store.read_attachment(owner, item['id'], found['offset'], 10)['pages'], [2])
        self.assertEqual(store.search_attachment(owner, item['id'], 'missing')['matches'], [])
        with self.assertRaises(ValueError): store.search_attachment(other, item['id'], 'Method')
        with self.assertRaises(ValueError): store.search_attachment(owner, item['id'], ' ')


class SessionDeletionTests(unittest.TestCase):
    def test_delete_removes_only_owned_uploads_and_record(self):
        with harness() as (app, fake, root, owner):
            store = app.app.agent.store
            (root/'keep.txt').write_text('project content')
            other = store.create()
            own = store.upload(owner['id'], 'own.txt', base64.b64encode(b'private upload').decode())
            keep = store.upload(other['id'], 'keep.txt', base64.b64encode(b'other upload').decode())
            status, raw = post(app, '/api/session/delete', {'session_id': owner['id']})
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)['removed_uploads'], 1)
            self.assertFalse(store.session_path(owner['id']).exists())
            self.assertFalse((store.root/'uploads'/own['id']).exists())
            self.assertTrue((store.root/'uploads'/keep['id']/'original').exists())
            self.assertEqual((root/'keep.txt').read_text(), 'project content')
            self.assertEqual([s['id'] for s in store.list_sessions()], [other['id']])
            with self.assertRaises(ValueError): store.load(owner['id'])
            self.assertEqual(post(app, '/api/session/delete', {'session_id': owner['id']})[0], 200)

    def test_shared_attachment_and_corrupt_other_session_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(d)
            owner, other = store.create(), store.create()
            shared = store.upload(owner['id'], 'shared.txt', base64.b64encode(b'shared').decode())
            other['attachments'] = [shared]
            store.save(other)
            self.assertEqual(store.delete(owner['id'])['removed_uploads'], 0)
            self.assertEqual(store.read_attachment(other, shared['id'])['text'], 'shared')
            corrupt = store.create()
            store.session_path(corrupt['id']).write_text('{broken')
            self.assertTrue(store.delete(other['id'])['warning'])
            self.assertTrue((store.root/'uploads'/shared['id']/'original').exists())

    def test_invalid_cross_origin_and_active_deletion_do_not_change_record(self):
        import http.client
        with harness() as (app, fake, root, session):
            for value in [None, '../sessions', {}, 'a'*31]:
                self.assertEqual(post(app, '/api/session/delete', {'session_id': value})[0], 400)
            for extra in [{'Origin': 'https://evil.example'}, {'Host': 'evil.example'}]:
                conn = http.client.HTTPConnection('127.0.0.1', app.server_port, timeout=3)
                conn.request('POST', '/api/session/delete', json.dumps({'session_id': session['id']}),
                             {'Content-Type': 'application/json', **extra})
                response = conn.getresponse()
                self.assertIn(response.status, (400, 403))
                response.read(); conn.close()
            active = app.app.begin('busy-delete-test')
            self.assertIsNotNone(active)
            self.assertEqual(post(app, '/api/session/delete', {'session_id': session['id']})[0], 409)
            self.assertTrue(app.app.agent.store.session_path(session['id']).exists())
            app.app.finish(active)


class AgentTests(unittest.TestCase):
    def test_chat_can_read_beyond_excerpt_and_reuse_attachment_on_followup(self):
        with harness() as (app, fake, root, session):
            text = 'Introduction. ' * 500 + '\nMethod: use target states.\nResults: verified.'
            item = app.app.agent.store.upload(session['id'], 'paper.txt', base64.b64encode(text.encode()).decode())
            request = body(session, 'chat')
            request['attachments'] = [item['id']]
            fake.plans.put(Plan(events=tool_events('read_attachment', {'attachment_id': item['id'], 'offset': 7000})))
            fake.plans.put(Plan(events=answer_events('方法使用目标模型状态。')))
            _, raw = post(app, '/api/agent', request)
            self.assertFalse([v for k, v in decode_events(raw) if k == 'error'])
            self.assertEqual({t['function']['name'] for t in fake.captures[0][1]['tools']},
                             {'read_attachment', 'search_attachment'})
            result = json.loads(fake.captures[1][1]['messages'][-1]['content'])
            self.assertIn('Method: use target states', result['result']['text'])
            followup = body(session, 'chat')
            followup['text'] = '继续解释实验'
            fake.plans.put(Plan(events=tool_events('search_attachment', {'attachment_id': item['id'], 'query': 'Results'})))
            fake.plans.put(Plan(events=answer_events('实验结果见附件。')))
            _, raw = post(app, '/api/agent', followup)
            self.assertFalse([v for k, v in decode_events(raw) if k == 'error'])
            self.assertIn(item['id'], fake.captures[2][1]['messages'][0]['content'])
            self.assertEqual(json.loads(fake.captures[3][1]['messages'][-1]['content'])['result']['matches'][0]['offset'], text.index('Results'))

    def test_prose_tool_call_requests_real_tool_instead_of_executing_code(self):
        with harness() as (app, fake, root, session):
            item = app.app.agent.store.upload(session['id'], 'paper.txt', base64.b64encode(b'Actual methods and results.').decode())
            fake.plans.put(Plan(events=answer_events('我将读取剩余内容。\n```python\nread_attachment(attachment_id="invented", length=10000)\n```')))
            fake.plans.put(Plan(events=tool_events('read_attachment', {'attachment_id': item['id']}, finish='stop')))
            fake.plans.put(Plan(events=answer_events('根据方法与结果，创新是……')))
            _, raw = post(app, '/api/agent', body(session, 'chat'))
            events = decode_events(raw)
            self.assertFalse([v for k, v in events if k == 'error'])
            actual = [v for k, v in events if k == 'tool' and v['status'] == 'complete']
            self.assertEqual(len(actual), 1)
            self.assertEqual(actual[0]['arguments'], {'attachment_id': item['id']})
            self.assertEqual(fake.captures[1][1]['tool_choice']['function']['name'], 'read_attachment')
            self.assertEqual(next(v for k, v in events if k == 'stats')['task_status'], 'complete')

    def test_repeated_pretend_call_is_not_marked_complete(self):
        with harness() as (app, fake, root, session):
            app.app.agent.store.upload(session['id'], 'paper.txt', base64.b64encode(b'paper').decode())
            for _ in range(2):
                fake.plans.put(Plan(events=answer_events('```python\nread_attachment(attachment_id="none")\n```')))
            _, raw = post(app, '/api/agent', body(session, 'chat'))
            events = decode_events(raw)
            self.assertTrue([v for k, v in events if k == 'error'])
            self.assertFalse([v for k, v in events if k in ('tool', 'stats')])
            self.assertEqual(app.app.agent.store.load(session['id'])['turns'][-1]['status'], 'error')

    def test_requested_tool_example_is_not_run(self):
        with harness() as (app, fake, root, session):
            app.app.agent.store.upload(session['id'], 'paper.txt', base64.b64encode(b'paper').decode())
            request = body(session, 'chat')
            request['text'] = '请给我 read_attachment 的代码示例'
            fake.plans.put(Plan(events=answer_events('```python\nread_attachment(attachment_id="example")\n```')))
            _, raw = post(app, '/api/agent', request)
            self.assertFalse([v for k, v in decode_events(raw) if k in ('tool', 'error')])
            self.assertEqual(len(fake.captures), 1)

    def test_token_budget_preserves_long_paper_and_prunes_complete_old_turns(self):
        with harness() as (app, fake, root, session):
            paper = 'History-free decoding methods and evaluation. ' * 1100
            messages = [{'role': 'user', 'content': 'Analyze paper'},
                        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'read', 'function': {'name': 'read_attachment', 'arguments': '{}'}}]},
                        {'role': 'tool', 'tool_call_id': 'read', 'content': paper}]
            notices = []
            kept = app.app.agent._context(messages, lambda *e: notices.append(e),
                                          lambda _: {'count': 15000, 'max_model_len': 32768})
            self.assertEqual(kept, messages)
            self.assertFalse(notices)
            messages.append({'role': 'user', 'content': 'New task'})
            kept = app.app.agent._context(messages, lambda *e: notices.append(e),
                                          lambda _: {'count': 32000, 'max_model_len': 32768})
            self.assertEqual(kept, [messages[-1]])
            self.assertTrue(notices)
            self.assertEqual(messages[2]['content'], paper)

    def test_length_limited_answer_continues_and_reports_actual_final_status(self):
        for continues, expected in [(1, 'complete'), (3, 'incomplete')]:
            with self.subTest(continues=continues), harness() as (app, fake, root, session):
                for i in range(continues):
                    fake.plans.put(Plan(events=answer_events('段落'+str(i), 'length')))
                if expected == 'complete':
                    fake.plans.put(Plan(events=answer_events('完整结论。')))
                _, raw = post(app, '/api/agent', body(session))
                events = decode_events(raw)
                self.assertFalse([v for k, v in events if k == 'error'])
                self.assertEqual(next(v for k, v in events if k == 'stats')['task_status'], expected)
                self.assertEqual(app.app.agent.store.load(session['id'])['turns'][-1]['status'], expected)
                later = [v for k, v in events if k == 'delta' and v['round'] > 1]
                self.assertTrue(later)
                self.assertTrue(all(v['continuation_of'] == 1 for v in later))
                self.assertEqual(fake.captures[1][1]['tool_choice'], 'none')
                self.assertEqual(sum(m['role'] == 'system' for m in fake.captures[1][1]['messages']), 1)
                self.assertTrue(fake.captures[1][1]['continue_final_message'])
                self.assertFalse(fake.captures[1][1]['add_generation_prompt'])
                self.assertEqual(fake.captures[0][1]['messages'][0], fake.captures[1][1]['messages'][0])
                saved = app.app.agent.store.load(session['id'])
                self.assertEqual(len(saved['messages']), 2)
                self.assertEqual(saved['messages'][-1]['content'],
                                 ''.join('段落'+str(i) for i in range(continues)) + ('完整结论。' if expected == 'complete' else ''))

    def test_continuation_can_end_immediately_after_last_content_token(self):
        with harness() as (app, fake, root, session):
            fake.plans.put(Plan(events=answer_events('结论已经完整。', 'length')))
            fake.plans.put(Plan(events=answer_events('', 'stop')))
            _, raw = post(app, '/api/agent', body(session, 'chat'))
            events = decode_events(raw)
            self.assertFalse([v for k, v in events if k == 'error'])
            self.assertEqual(next(v for k, v in events if k == 'stats')['task_status'], 'complete')
            self.assertEqual(app.app.agent.store.load(session['id'])['messages'][-1]['content'], '结论已经完整。')

    def test_examples_prepare_real_inputs_without_overwriting_existing_work(self):
        with harness() as (app, fake, root, session):
            status, raw = post(app, '/api/example', {'session_id': session['id'], 'case_id': 'project'})
            self.assertEqual(status, 200, raw)
            self.assertTrue((root/'harness-demo/brief.json').is_file())
            (root/'harness-demo/brief.json').write_text('user changes')
            post(app, '/api/example', {'session_id': session['id'], 'case_id': 'project'})
            self.assertEqual((root/'harness-demo/brief.json').read_text(), 'user changes')
            status, raw = post(app, '/api/example', {'session_id': session['id'], 'case_id': 'multimodal'})
            self.assertEqual(status, 200, raw)
            data = json.loads(raw)
            self.assertEqual([a['kind'] for a in data['attachments']], ['image', 'document'])
            self.assertNotIn('250', data['prompt'])
            self.assertIn('250', data['check'])
            status, raw = post(app, '/api/example', {'session_id': session['id'], 'case_id': 'search'})
            self.assertEqual(json.loads(raw)['mode'], 'search')

    def test_foreign_origin_and_rebinding_host_cannot_write_files(self):
        import http.client
        with harness() as (app, fake, root, session):
            for extra in [{'Origin': 'https://evil.example'}, {'Host': 'evil.example'}]:
                conn = http.client.HTTPConnection('127.0.0.1', app.server_port, timeout=3)
                conn.request('POST', '/api/file', json.dumps({'path': 'evil.txt', 'content': 'bad',
                             'expected_sha256': 'new'}), {'Content-Type': 'application/json', **extra})
                response = conn.getresponse()
                self.assertIn(response.status, (400, 403))
                response.read(); conn.close()
                self.assertFalse((root/'evil.txt').exists())

    def test_fragmented_tool_calls_read_edit_and_final_response(self):
        with harness() as (app, fake, root, session):
            p=root/'demo.txt';p.write_text('old content\n')
            fake.plans.put(Plan(events=tool_events('read_file', {'path':'demo.txt'})))
            fake.plans.put(Plan(events=tool_events('edit_file', {'path':'demo.txt', 'old_text':'old',
                               'new_text':'new', 'expected_sha256':sha(p.read_bytes())})))
            fake.plans.put(Plan(events=chat_events()))
            status, raw = post(app, '/api/agent', body(session))
            events = decode_events(raw)
            self.assertEqual(status, 200)
            self.assertFalse([v for k,v in events if k=='error'], events)
            self.assertEqual(p.read_text(), 'new content\n')
            self.assertEqual(len(fake.captures), 3)
            third = fake.captures[2][1]['messages']
            self.assertEqual([m['role'] for m in third], ['system','user','assistant','tool','assistant','tool'])
            self.assertIn('sha256', third[3]['content'])
            stats = next(v for k,v in events if k=='stats')
            self.assertEqual(stats['model_rounds'], 3)
            self.assertEqual(stats['tool_calls'], 2)
            self.assertEqual(stats['metric_source'], 'server')
            saved = app.app.agent.store.load(session['id'])
            self.assertEqual(saved['turns'][-1]['status'], 'complete')
            self.assertEqual(len(saved['messages']), 6)

    def test_wrong_mode_and_unknown_arguments_become_tool_errors(self):
        for name, args in [('write_file', {'path':'bad.txt','content':'x','expected_sha256':'new'}),
                           ('read_file', {'path':'file.txt','shell':'rm -rf /'}),
                           ('read_file', {'path': float('nan')})]:
            with self.subTest(name=name), harness() as (app,fake,root,session):
                fake.plans.put(Plan(events=tool_events(name,args)))
                fake.plans.put(Plan(events=chat_events()))
                _,raw=post(app,'/api/agent',body(session,'search'))
                result=[v for k,v in decode_events(raw) if k=='tool' and v['status']=='error']
                self.assertTrue(result)
                self.assertFalse((root/'bad.txt').exists())
                provided={t['function']['name'] for t in fake.captures[0][1]['tools']}
                self.assertIn('web_search',provided)
                self.assertNotIn('write_file',provided)

    def test_truncated_or_incomplete_tools_never_execute(self):
        for finish,done in [('length',True),('tool_calls',False),('stop',True)]:
            with self.subTest(finish=finish,done=done), harness() as (app,fake,root,session):
                fake.plans.put(Plan(events=tool_events('write_file',{'path':'unsafe.txt','content':'x','expected_sha256':'new'},finish,done)))
                _,raw=post(app,'/api/agent',body(session))
                events=decode_events(raw)
                self.assertTrue([v for k,v in events if k=='error'])
                self.assertFalse((root/'unsafe.txt').exists())
                self.assertFalse([v for k,v in events if k=='stats'])

    def test_final_round_disables_tools_and_preserves_completed_write(self):
        with harness() as (app,fake,root,session):
            app.app.config.max_agent_steps=2
            fake.plans.put(Plan(events=tool_events('write_file',{'path':'done.txt','content':'saved','expected_sha256':'new'})))
            fake.plans.put(Plan(events=chat_events()))
            _,raw=post(app,'/api/agent',body(session))
            self.assertFalse([v for k,v in decode_events(raw) if k=='error'])
            self.assertNotIn('tools', fake.captures[1][1])
            self.assertNotIn('tool_choice', fake.captures[1][1])
            self.assertEqual([m['role'] for m in fake.captures[1][1]['messages']].count('system'), 1)
            self.assertEqual((root/'done.txt').read_text(),'saved')

    def test_repeated_web_failures_finish_with_tool_free_answer(self):
        with harness() as (app, fake, root, session):
            for i in range(3):
                fake.plans.put(Plan(events=tool_events('web_search', {'query': f'query {i}'})))
            fake.plans.put(Plan(events=answer_events('搜索来源暂不可访问，无法核实具体规格。')))
            with patch('agent.search_web', side_effect=ValueError('provider unavailable')):
                _, raw = post(app, '/api/agent', body(session, 'search'))
            self.assertFalse([v for k,v in decode_events(raw) if k == 'error'])
            self.assertEqual(len(fake.captures), 4)
            self.assertNotIn('tools', fake.captures[-1][1])
            self.assertIn('当前日期：', fake.captures[0][1]['messages'][0]['content'])
            self.assertEqual(app.app.agent.store.load(session['id'])['turns'][-1]['status'], 'complete')

    def test_empty_model_answer_preserves_sources_and_is_not_marked_complete(self):
        with harness() as (app, fake, root, session):
            fake.plans.put(Plan(events=tool_events('fetch_page', {'url': 'https://example.com/spec'})))
            fake.plans.put(Plan(events=answer_events('')))
            with patch('agent.fetch_page', return_value={'title': 'Official product', 'url': 'https://example.com/spec',
                                                        'text': 'Product 495: 16 cores'}):
                _, raw = post(app, '/api/agent', body(session, 'search'))
            events = decode_events(raw)
            self.assertFalse([v for k,v in events if k == 'error'])
            report = next(v for k,v in events if k == 'delta' and v.get('origin') == 'harness')
            self.assertIn('工作台提示', report['text'])
            self.assertIn('https://example.com/spec', report['text'])
            self.assertIn('16 cores', report['text'])
            saved = app.app.agent.store.load(session['id'])
            self.assertEqual(saved['turns'][-1]['status'], 'incomplete')
            self.assertEqual(saved['messages'][-1]['content'], report['text'])

    def test_cancel_after_write_preserves_outcome_and_releases_slot(self):
        with harness() as (app,fake,root,session):
            fake.plans.put(Plan(events=tool_events('write_file',{'path':'saved.txt','content':'saved','expected_sha256':'new'})))
            hold=Plan(events=[],hold=True);fake.plans.put(hold)
            result=[]
            thread=threading.Thread(target=lambda:result.append(post(app,'/api/agent',body(session))))
            thread.start();self.assertTrue(hold.started.wait(4))
            post(app,'/api/cancel',{'request_id':'test-agent'})
            thread.join(4);self.assertFalse(thread.is_alive())
            self.assertEqual((root/'saved.txt').read_text(),'saved')
            self.assertIsNone(app.app.active)
            saved=app.app.agent.store.load(session['id'])
            self.assertEqual(saved['turns'][-1]['status'],'cancelled')
            self.assertTrue(any(m['role']=='tool' and 'saved.txt' in m['content'] for m in saved['messages']))

    def test_multimodal_chat_without_tools_and_persisted_context(self):
        from PIL import Image
        with harness() as (app,fake,root,session):
            out=io.BytesIO();Image.new('RGB',(32,32),'blue').save(out,'PNG')
            item=app.app.agent.store.upload(session['id'],'image.png',base64.b64encode(out.getvalue()).decode())
            b=body(session,'chat');b['attachments']=[item['id']]
            fake.plans.put(Plan(events=chat_events()))
            _,raw=post(app,'/api/agent',b)
            self.assertFalse([v for k,v in decode_events(raw) if k=='error'])
            request=fake.captures[0][1]
            self.assertNotIn('tools',request)
            self.assertTrue(request['messages'][1]['content'][2]['image_url']['url'].startswith('data:image/jpeg;base64,'))
            loaded=app.app.agent.store.load(session['id'])
            self.assertEqual(len(loaded['attachments']),1)
            self.assertEqual(len(loaded['messages']),2)


class WebBoundaryTests(unittest.TestCase):
    def test_weather_uses_resolved_city_units_and_timestamps(self):
        geo = {'results': [{'name': '北京', 'admin1': '北京市', 'country': '中国',
                            'latitude': 39.9, 'longitude': 116.4}]}
        forecast = {'timezone': 'Asia/Shanghai',
                    'current': {'time': '2026-10-27T10:30', 'temperature_2m': 10.8, 'weather_code': 0},
                    'current_units': {'temperature_2m': '°C'},
                    'daily': {'time': ['2026-10-27'], 'weather_code': [3],
                              'temperature_2m_max': [14], 'temperature_2m_min': [7],
                              'precipitation_probability_max': [0]}}
        with patch('web_tools.fetch_page', side_effect=[{'text': json.dumps(geo)}, {'text': json.dumps(forecast)}]) as get:
            result = get_weather('北京', 1)
            self.assertEqual(result['current']['description'], '晴')
            self.assertEqual(result['forecast'][0]['description'], '阴')
            self.assertEqual(result['timezone'], 'Asia/Shanghai')
            self.assertIn('北京市', result['location'])
            self.assertIn('latitude=39.9', get.call_args_list[1].args[0])
            self.assertEqual(result['source_url'], 'https://open-meteo.com/')
        with patch('web_tools.fetch_page', return_value={'text': '{"results":[]}'}):
            with self.assertRaises(ValueError): get_weather('unknown city')
        for days in [0, 8, True]:
            with self.assertRaises(ValueError): get_weather('北京', days)

    def test_private_ipv4_ipv6_and_mixed_dns_rejected(self):
        for ip in ['127.0.0.1','10.1.2.3','169.254.169.254','::1','::ffff:127.0.0.1']:
            with self.subTest(ip=ip), patch('web_tools.socket.getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',(ip,443))]):
                with self.assertRaises(ValueError): public_address('https://example.com')
        for url in ['file:///etc/passwd','http://user:password@example.com','http://example.com:8000']:
            with self.assertRaises(ValueError): public_address(url)

    def test_safe_public_address_is_pinned(self):
        addr=(socket.AF_INET,socket.SOCK_STREAM,6,'',('93.184.216.34',443))
        with patch('web_tools.socket.getaddrinfo',return_value=[addr]) as lookup:
            p, result=public_address('https://example.com/path')
            self.assertEqual(result,addr)
            self.assertEqual(lookup.call_count,1)


if __name__=='__main__':
    unittest.main()
