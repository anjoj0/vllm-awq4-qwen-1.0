"""Search fallbacks and bounded, query-focused extraction; no public network."""
import base64
from html import escape
import json
import unittest
from unittest.mock import patch

from web_tools import extract_html, parse_search, search_providers, search_variants, ranked_results


class SearchTests(unittest.TestCase):
    def test_bing_html_links_are_resolved_to_actual_sources(self):
        source = 'https://vendor.example/products/495'
        encoded = 'a1'+base64.urlsafe_b64encode(source.encode()).decode().rstrip('=')
        raw = f'''<html><ol><li class="b_algo"><h2><a href="https://www.bing.com/ck/a?u={encoded}">Product 495</a></h2>
          <div class="b_caption"><p>16 cores &amp; 32 threads</p></div></li></ol></html>'''
        rows = parse_search(raw, 'bing', 'https://www.bing.com/search?q=495')
        self.assertEqual(rows[0]['url'], source)
        self.assertEqual(rows[0]['snippet'], '16 cores & 32 threads')

    def test_blocked_provider_and_broadened_results_use_focused_fallback(self):
        wrong = {'title': 'Acme 395 specifications', 'url': 'https://vendor.example/395', 'snippet': 'older model'}
        right = {'title': 'Acme 495 specifications', 'url': 'https://vendor.example/495', 'snippet': 'new model'}
        with patch('web_tools.fetch_search', side_effect=[[wrong], OSError('provider blocked'), [right]]) as fetch:
            result = search_providers('Acme processor 495 specifications')
        self.assertEqual(result['results'], [right])
        self.assertEqual(result['relevance'], 'matched')
        self.assertEqual(result['effective_query'], '"495" "acme"')
        self.assertEqual(fetch.call_count, 3)
        self.assertEqual([a['status'] for a in result['attempts']], ['weak_results', 'unavailable', 'matched'])

    def test_general_query_can_fall_back_to_sogou(self):
        row = {'title': 'prefix caching guide', 'url': 'https://docs.example/guide', 'snippet': 'prefix caching'}
        with patch('web_tools.fetch_search', side_effect=[OSError('offline'), [row]]):
            result = search_providers('prefix caching')
        self.assertEqual(result['provider'], 'sogou')
        self.assertEqual(result['results'], [row])

    def test_empty_search_is_explicit_and_site_constraint_is_preserved(self):
        with patch('web_tools.fetch_search', return_value=[]), self.assertRaisesRegex(ValueError, '多个搜索来源'):
            search_providers('unknown')
        rows = [dict(title='Acme 495', url='https://evil.example/495', snippet='Acme 495'),
                dict(title='Acme 495', url='https://www.vendor.example/495', snippet='Acme 495')]
        ranked, matched = ranked_results(rows, 'Acme 495 site:vendor.example')
        self.assertTrue(matched)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]['url'], rows[1]['url'])
        self.assertEqual(search_variants('Acme 495 specs 2026 site:vendor.example')[1], 'site:vendor.example "495" "acme"')


class PageExtractionTests(unittest.TestCase):
    def test_large_embedded_catalog_matches_product_not_sort_metadata(self):
        def product(title, weight, cores):
            return {'title': title, 'sortWeight': weight,
                    'productPages': {'en': 'https://vendor.example/'+str(cores)},
                    'elements': {'cores': {'title': 'CPU cores', 'formatValue': str(cores)},
                                 'tdp': {'title': 'Default TDP', 'formatValue': '55', 'suffix': 'W'}}}
        data = {'items': [product('Acme Max 395', 4950, 8), product('Acme™ AI Max+ 495', 9999, 16)],
                'padding': 'x'*(2*1024*1024)}
        raw = '<html><title>Catalog</title><main><div data-json="'+escape(json.dumps(data), quote=True)+'"></div></main></html>'
        result = extract_html(raw, 'https://vendor.example', 'AI Max 495')
        self.assertEqual(result['structured_matches'], 1)
        self.assertIn('CPU cores: 16', result['text'])
        self.assertIn('Default TDP: 55W', result['text'])
        self.assertNotIn('Max 395', result['text'])
        self.assertEqual(result['links'][0]['url'], 'https://vendor.example/16')
        self.assertLessEqual(len(result['text']), 12000)

    def test_keyword_selects_late_body_and_table_labels_remain_readable(self):
        raw = '<html><main><p>'+'unrelated introduction '*2000+'</p><h2>Acme 495</h2><table><tr><th>Cores</th><td>16</td></tr></table></main></html>'
        result = extract_html(raw, 'https://vendor.example/spec', 'Acme 495')
        self.assertIn('Acme 495', result['text'])
        self.assertIn('Cores 16', result['text'])
        self.assertTrue(result['truncated'])
        self.assertLessEqual(len(result['text']), 12000)

    def test_jsonld_and_plain_html_without_query(self):
        raw = '''<html><title>Guide</title><script>fetch('/unsafe')</script>
          <script type="application/ld+json">{"@type":"Product","name":"Acme 495","description":"16 cores","url":"/495"}</script>
          <nav>Menu</nav><main><h1>Guide</h1><p>Actual body</p><a href="/reference">Reference</a></main></html>'''
        result = extract_html(raw, 'https://vendor.example/guide')
        self.assertIn('Actual body', result['text'])
        self.assertIn('16 cores', result['text'])
        self.assertNotIn('unsafe', result['text'])
        self.assertNotIn('Menu', result['text'])
        self.assertEqual(result['links'][0]['url'], 'https://vendor.example/495')


if __name__ == '__main__':
    unittest.main()
