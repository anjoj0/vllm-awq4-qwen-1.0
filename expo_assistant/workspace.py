"""Bounded project operations, attachment ingestion, and local conversation storage."""
from __future__ import annotations

import base64
import binascii
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from xml.etree import ElementTree

MAX_FILE = 1024 * 1024
MAX_UPLOAD = 12 * 1024 * 1024
BLOCKED = {'.git', '.runtime', '.ssh', '.aws', '.gnupg', '.venv', 'venv',
           'node_modules', '__pycache__', '.hf-cache', '.triton-cache', '.vllm-cache'}
SECRET_SUFFIXES = {'.pem', '.key', '.p12', '.pfx', '.safetensors', '.gguf', '.bin'}


def opaque_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{32}', value):
        raise ValueError('无效的会话或附件 ID')
    return value


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic_write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.harness-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(name, path.stat().st_mode & 0o777 if path.exists() else 0o600)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Workspace:
    def __init__(self, root, data_dir):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError('工作区必须是目录')
        self.data_dir = Path(data_dir).resolve()
        self.lock = threading.RLock()

    def path(self, name, *, directory=False):
        if not isinstance(name, str) or len(name) > 500 or '\x00' in name:
            raise ValueError('路径格式不正确')
        rel = Path(name or '.')
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('请使用工作区内的相对路径，不能越出工作区')
        for part in rel.parts:
            if (part in BLOCKED or part.startswith(('.env', '.triton-cache', '.vllm-cache', '.venv'))
                    or Path(part).suffix.lower() in SECRET_SUFFIXES):
                raise ValueError('该路径不向工具开放')
        target = self.root / rel
        # Reject every symlink component, including links that currently resolve inside root.
        walk = self.root
        for part in rel.parts:
            walk = walk / part
            if walk.is_symlink():
                raise ValueError('文件工具不跟随符号链接')
        target = target.resolve()
        if not target.is_relative_to(self.root) or target.is_relative_to(self.data_dir):
            raise ValueError('该路径不在可操作的项目范围内')
        if target == self.root and not directory:
            raise ValueError('请指定文件')
        return target

    def list_files(self, path='.'):
        target = self.path(path, directory=True)
        if not target.is_dir():
            raise ValueError('目录不存在')
        entries = []
        for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            try:
                checked = self.path(str(item.relative_to(self.root)), directory=True)
            except ValueError:
                continue
            if not (checked.is_file() or checked.is_dir()):
                continue
            entries.append({'name': item.name, 'path': str(item.relative_to(self.root)),
                            'kind': 'directory' if item.is_dir() else 'file',
                            'size': item.stat().st_size if item.is_file() else None})
            if len(entries) >= 200:
                break
        return {'path': str(target.relative_to(self.root)), 'entries': entries,
                'limited': len(entries) >= 200}

    def _read(self, path):
        target = self.path(path)
        if not target.is_file() or target.stat().st_size > MAX_FILE:
            raise ValueError('只支持读取 1 MiB 以内的普通文本文件')
        with target.open('rb') as stream:
            raw = stream.read(MAX_FILE + 1)
        if len(raw) > MAX_FILE or b'\0' in raw:
            raise ValueError('文件过大或为二进制，请通过附件上传解析')
        try:
            return target, raw, raw.decode('utf-8-sig')
        except UnicodeError as exc:
            raise ValueError('项目文件须为 UTF-8 文本；文档请通过附件上传') from exc

    def read_file(self, path, start_line=1, max_lines=160):
        if type(start_line) is not int or start_line < 1 or type(max_lines) is not int or not 1 <= max_lines <= 400:
            raise ValueError('行号须为正整数，每次可读 1–400 行')
        _, raw, content = self._read(path)
        lines = content.splitlines(keepends=True)
        text = ''.join(lines[start_line-1:start_line-1+max_lines])
        truncated = len(text) > 16000
        return {'path': path, 'sha256': sha(raw), 'start_line': start_line,
                'total_lines': len(lines), 'content': text[:16000],
                'truncated': truncated or start_line - 1 + max_lines < len(lines)}

    def write_file(self, path, content, expected_sha256):
        if not isinstance(content, str) or len(content.encode('utf-8')) > MAX_FILE or '\0' in content:
            raise ValueError('写入内容须为 1 MiB 以内的文本')
        if not isinstance(expected_sha256, str):
            raise ValueError('必须提供读取时的 sha256，新文件填 new')
        with self.lock:
            target = self.path(path)
            exists = target.exists()
            old = b''
            if exists:
                _, old, _ = self._read(path)
                if expected_sha256 != sha(old):
                    raise ValueError('文件已改变或尚未读取。请重新 read_file 获取 sha256 后再修改')
            elif expected_sha256 != 'new':
                raise ValueError('文件不存在；新文件 expected_sha256 应为 new')
            raw = content.encode('utf-8')
            operation = uuid.uuid4().hex
            backup_dir = self.data_dir / 'changes' / operation
            backup_dir.mkdir(parents=True, mode=0o700)
            if exists:
                atomic_write(backup_dir / 'before', old)
            diff = ''.join(difflib.unified_diff(old.decode('utf-8-sig').splitlines(True),
                         content.splitlines(True), fromfile=path+' (before)', tofile=path+' (after)'))
            atomic_write(backup_dir / 'change.json', json.dumps({'path': path,
                         'created': not exists, 'before_sha256': sha(old) if exists else None,
                         'after_sha256': sha(raw), 'time': time.time(), 'diff': diff},
                         ensure_ascii=False).encode())
            atomic_write(target, raw)
            return {'path': path, 'sha256': sha(raw), 'bytes': len(raw), 'created': not exists,
                    'change_id': operation, 'diff': diff[:12000], 'diff_truncated': len(diff) > 12000}

    def edit_file(self, path, old_text, new_text, expected_sha256):
        if not isinstance(old_text, str) or not old_text or not isinstance(new_text, str):
            raise ValueError('old_text 不能为空，替换内容须为文本')
        with self.lock:
            _, raw, content = self._read(path)
            if sha(raw) != expected_sha256:
                raise ValueError('文件已改变，请重新读取')
            if content.count(old_text) != 1:
                raise ValueError('old_text 必须在文件中恰好出现一次，请扩大匹配片段')
            return self.write_file(path, content.replace(old_text, new_text, 1), expected_sha256)

    def search_files(self, query, path='.', cancelled=None):
        if not isinstance(query, str) or not 1 <= len(query) <= 200:
            raise ValueError('搜索词须为 1–200 字符，按字面匹配')
        start = self.path(path, directory=True)
        if not start.is_dir():
            raise ValueError('搜索起点须为目录')
        hits, scanned, total = [], 0, 0
        for folder, dirs, files in os.walk(start, followlinks=False):
            allowed = []
            for name in sorted(dirs):
                try:
                    self.path(str((Path(folder)/name).relative_to(self.root)), directory=True)
                    allowed.append(name)
                except ValueError:
                    pass
            dirs[:] = allowed
            for name in sorted(files):
                if cancelled and cancelled.is_set():
                    raise InterruptedError('cancelled')
                relative = str((Path(folder)/name).relative_to(self.root))
                try:
                    _, raw, content = self._read(relative)
                except (ValueError, OSError):
                    continue
                scanned += 1
                total += len(raw)
                for number, line in enumerate(content.splitlines(), 1):
                    if query.casefold() in line.casefold():
                        hits.append({'path': relative, 'line': number, 'text': line[:400]})
                        if len(hits) >= 40:
                            return {'matches': hits, 'scanned_files': scanned, 'limited': True}
                if scanned >= 2000 or total > 32 * MAX_FILE:
                    return {'matches': hits, 'scanned_files': scanned, 'limited': True}
        return {'matches': hits, 'scanned_files': scanned, 'limited': False}


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()

    def session_path(self, session_id):
        return self.root / 'sessions' / (opaque_id(session_id) + '.json')

    def create(self):
        obj = {'id': uuid.uuid4().hex, 'title': '新对话', 'updated': time.time(),
               'messages': [], 'turns': [], 'attachments': []}
        self.save(obj)
        return obj

    def load(self, session_id):
        try:
            with self.lock:
                return json.loads(self.session_path(session_id).read_text())
        except FileNotFoundError as exc:
            raise ValueError('会话不存在，请新建对话') from exc

    def save(self, session):
        with self.lock:
            session['updated'] = time.time()
            atomic_write(self.session_path(session['id']), json.dumps(session, ensure_ascii=False).encode())

    def list_sessions(self):
        items = []
        for path in (self.root/'sessions').glob('*.json'):
            try:
                obj = json.loads(path.read_text())
                items.append({k: obj[k] for k in ('id', 'title', 'updated')})
            except (ValueError, KeyError, OSError):
                continue
        return sorted(items, key=lambda s: s['updated'], reverse=True)[:60]

    def delete(self, session_id):
        """Remove a conversation and its exclusive uploads, never workspace files."""
        with self.lock:
            path = self.session_path(session_id)
            if not path.exists():
                return {'deleted': True, 'session_id': session_id, 'removed_uploads': 0}
            session = self.load(session_id)
            owned = {opaque_id(a['id']) for a in session.get('attachments', [])}
            shared, warnings = set(), []
            # Inspect all records, including those beyond the sidebar's display limit.
            for other in path.parent.glob('*.json'):
                if other == path:
                    continue
                try:
                    record = json.loads(other.read_text())
                    shared.update(opaque_id(a['id']) for a in record.get('attachments', []))
                except (ValueError, KeyError, TypeError, AttributeError, OSError):
                    shared.update(owned)
                    warnings.append('部分会话无法读取，已保留可能共用的附件。')
            path.unlink()
            removed = 0
            for attachment_id in owned - shared:
                folder = self.root / 'uploads' / attachment_id
                try:
                    if folder.is_symlink():
                        folder.unlink()
                    elif folder.exists():
                        shutil.rmtree(folder)
                    else:
                        continue
                    removed += 1
                except OSError:
                    warnings.append('会话已删除，但部分附件文件未能清理。')
            return {'deleted': True, 'session_id': session_id, 'removed_uploads': removed,
                    'warning': ' '.join(dict.fromkeys(warnings))}

    def public_session(self, session):
        return {k: session[k] for k in ('id', 'title', 'updated', 'turns', 'attachments')}

    def attachment(self, session, attachment_id):
        opaque_id(attachment_id)
        for item in session['attachments']:
            if item['id'] == attachment_id:
                return item
        raise ValueError('附件不属于此会话')

    def upload(self, session_id, name, encoded):
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValueError('文件名须为 1–200 字符')
        name = Path(name.replace('\\', '/')).name
        if not isinstance(encoded, str) or len(encoded) > MAX_UPLOAD * 4 // 3 + 8:
            raise ValueError('单个附件上限 12 MiB')
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError('附件编码无效') from exc
        if not raw or len(raw) > MAX_UPLOAD:
            raise ValueError('附件为空或超过 12 MiB')
        with self.lock:
            session = self.load(session_id)
            if len(session['attachments']) >= 24 or sum(a['size'] for a in session['attachments']) + len(raw) > 64 * MAX_FILE:
                raise ValueError('每个会话最多 24 个附件、总计 64 MiB，请新建对话')
            attachment_id = uuid.uuid4().hex
            folder = self.root / 'uploads' / attachment_id
            folder.mkdir(parents=True, mode=0o700)
            try:
                info, text, image = parse_attachment(name, raw)
                obj = {'id': attachment_id, 'name': name, 'size': len(raw), **info}
                atomic_write(folder/'original', raw)
                if text is not None:
                    atomic_write(folder/'text', text.encode())
                    obj['text_chars'] = len(text)
                    obj['excerpt'] = text[:300]
                if image is not None:
                    atomic_write(folder/'image.jpg', image)
                atomic_write(folder/'meta.json', json.dumps(obj, ensure_ascii=False).encode())
                session['attachments'].append(obj)
                self.save(session)
                return obj
            except Exception:
                shutil.rmtree(folder)
                raise

    def read_attachment(self, session, attachment_id, offset=0, limit=10000):
        item = self.attachment(session, attachment_id)
        if item['kind'] == 'image':
            raise ValueError('图片已通过视觉接口提供，请直接理解图片')
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 16000:
            raise ValueError('offset 须为非负整数，limit 须在 1–16000')
        text = (self.root/'uploads'/attachment_id/'text').read_text()
        page_starts = [(m.start(), int(m.group(1))) for m in re.finditer(r'--- 第 (\d+) 页 ---', text)]
        end = min(offset+limit, len(text))
        pages = [page for i, (start, page) in enumerate(page_starts)
                 if start < end and (page_starts[i+1][0] if i+1 < len(page_starts) else len(text)) > offset]
        return {'id': attachment_id, 'name': item['name'], 'text': text[offset:offset+limit],
                'offset': offset, 'next_offset': end, 'pages': pages,
                'total_chars': len(text), 'truncated': offset+limit < len(text),
                'note': item.get('note', '')}

    def search_attachment(self, session, attachment_id, query, max_matches=8):
        item = self.attachment(session, attachment_id)
        if item['kind'] == 'image':
            raise ValueError('图片由视觉接口理解，不能按文字搜索')
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 200:
            raise ValueError('附件搜索词须为 1–200 字符')
        if type(max_matches) is not int or not 1 <= max_matches <= 12:
            raise ValueError('max_matches 须在 1–12')
        text = (self.root/'uploads'/attachment_id/'text').read_text()
        matches = []
        for match in re.finditer(re.escape(query.strip()), text, re.IGNORECASE):
            start, end = max(0, match.start()-220), min(len(text), match.end()+450)
            pages = re.findall(r'--- 第 (\d+) 页 ---', text[:match.start()])
            matches.append({'offset': match.start(), 'page': int(pages[-1]) if pages else None,
                            'excerpt_offset': start, 'excerpt': text[start:end]})
            if len(matches) >= max_matches:
                break
        return {'id': attachment_id, 'name': item['name'], 'query': query,
                'matches': matches, 'total_chars': len(text),
                'note': '这些是匹配片段，可用 read_attachment(offset=excerpt_offset, limit=...) 读取上下文'}

    def document_inventory(self, session):
        return [{'attachment_id': a['id'], 'name': a['name'], 'total_chars': a.get('text_chars'),
                 'note': a.get('note', '')} for a in session['attachments'] if a['kind'] == 'document']

    def user_content(self, session, text, ids):
        parts = [{'type': 'text', 'text': text}]
        for attachment_id in ids:
            item = self.attachment(session, attachment_id)
            label = '\n附件 '+item['name']+'（id='+attachment_id+'）：'
            if item['kind'] == 'image':
                encoded = base64.b64encode((self.root/'uploads'/attachment_id/'image.jpg').read_bytes()).decode()
                parts.extend([{'type': 'text', 'text': label},
                              {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+encoded}}])
            else:
                data = self.read_attachment(session, attachment_id, 0, 6000)
                parts.append({'type': 'text', 'text': label+'\n'+json.dumps(data, ensure_ascii=False)})
        return text if not ids else parts


def parse_attachment(name, raw):
    try:
        return _parse_attachment(name, raw)
    except (ValueError, ImportError):
        raise
    except Exception as exc:
        raise ValueError('附件解析失败，请检查文件格式或重新导出：'+str(exc)[:180]) from exc


def _parse_attachment(name, raw):
    suffix = Path(name).suffix.lower()
    if suffix in {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'}:
        from PIL import Image, ImageOps
        try:
            with Image.open(io.BytesIO(raw)) as im:
                if im.width * im.height > 25_000_000:
                    raise ValueError('图片最多 2500 万像素')
                im.seek(0)
                im = ImageOps.exif_transpose(im).convert('RGB')
                original_size = list(im.size)
                im.thumbnail((1280, 1280))
                out = io.BytesIO()
                im.save(out, format='JPEG', quality=90)
                return {'kind': 'image', 'mime': 'image/jpeg', 'dimensions': list(im.size),
                        'original_dimensions': original_size, 'note': '视觉输入长边最多 1280 像素；动图取首帧'}, None, out.getvalue()
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError('无法解码此图片') from exc
    note = ''
    if suffix == '.pdf':
        from pypdf import PdfReader
        if not raw.startswith(b'%PDF-'):
            raise ValueError('文件不是有效的 PDF')
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise ValueError('暂不支持加密 PDF，请上传解密副本')
        if len(reader.pages) > 150:
            raise ValueError('PDF 最多 150 页，请拆分后上传')
        chunks, size = [], 0
        for i, page in enumerate(reader.pages):
            # Large content streams can inflate to hundreds of MB during text extraction.
            contents = page.get_contents()
            if contents is not None and len(contents.get_data()) > 8 * MAX_FILE:
                raise ValueError('PDF 页面内容过大，请拆分或导出为文本')
            part = '\n--- 第 '+str(i+1)+' 页 ---\n'+(page.extract_text() or '')
            chunks.append(part)
            size += len(part)
            if size >= 200000:
                note = '提取达到 20 万字符上限，后续页面未载入'
                break
        text = '\n'.join(chunks)
        if not any((c.isalnum() for c in re.sub(r'--- 第 \d+ 页 ---', '', text))):
            raise ValueError('此 PDF 没有可提取文字。扫描件请将所需页面作为图片上传进行视觉理解')
        note = note or 'PDF 提取文字；版面、图表请补充上传页面图片'
    elif suffix in {'.docx', '.xlsx', '.pptx'}:
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if sum(x.file_size for x in archive.infolist()) > 32 * MAX_FILE:
                    raise ValueError('文档解压后超过 32 MiB')
                if suffix == '.docx':
                    paths = ['word/document.xml']
                elif suffix == '.pptx':
                    paths = sorted((p for p in archive.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', p)),
                                   key=lambda p: int(re.search(r'slide(\d+)\.xml', p)[1]))
                else:
                    paths = sorted(p for p in archive.namelist() if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', p))
                shared = []
                if suffix == '.xlsx' and 'xl/sharedStrings.xml' in archive.namelist():
                    root = ElementTree.fromstring(archive.read('xl/sharedStrings.xml'))
                    shared = [''.join(n.itertext()) for n in root]
                chunks = []
                for p in paths[:150]:
                    root = ElementTree.fromstring(archive.read(p))
                    chunks.append('\n--- '+p+' ---\n')
                    if suffix == '.xlsx':
                        for row in root.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row'):
                            cells = []
                            for cell in row:
                                value = ''.join(cell.itertext())
                                if cell.get('t') == 's' and value.isdigit():
                                    value = shared[int(value)]
                                cells.append(cell.get('r', '')+': '+value)
                            chunks.append(' | '.join(cells))
                    else:
                        for paragraph in root.iter():
                            if paragraph.tag.rsplit('}', 1)[-1] == 'p':
                                chunks.append(''.join(n.text or '' for n in paragraph.iter() if n.tag.rsplit('}', 1)[-1] == 't'))
                text = '\n'.join(chunks)
                note = '提取文档文字与单元格值，不执行宏，不计算公式，不解析嵌入图片'
        except (zipfile.BadZipFile, ElementTree.ParseError, KeyError, IndexError) as exc:
            raise ValueError('Office 文档结构无效') from exc
    else:
        if suffix in SECRET_SUFFIXES or suffix in {'.exe', '.zip', '.tar', '.gz', '.7z', '.doc', '.xls', '.ppt'} or b'\0' in raw:
            raise ValueError('支持图片、PDF、DOCX/XLSX/PPTX 和 UTF-8 文本/代码文件')
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeError as exc:
            raise ValueError('文本请使用 UTF-8 编码；二进制文件暂不支持') from exc
    if len(text) > 200000:
        text = text[:200000]
        note += '；文本截取前 20 万字符'
    return {'kind': 'document', 'mime': 'text/plain', 'note': note}, text, None
