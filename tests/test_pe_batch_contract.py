import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
from validate_pe_tokenizer import build_development_batch, validate_development_batch
from transformers import AutoTokenizer


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize and not add_generation_prompt
        return '<user>' + messages[0]['content'] + '</user>'

    def __call__(self, text):
        # Stable fixture IDs preserve whitespace/codepoints and make the capacity case measurable.
        return {'input_ids': list(text.encode('utf-8'))}


class PeBatchContractTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.manifest = build_development_batch(self.tokenizer, {'tokenizer.json': {'sha256': 'a' * 64}})

    def test_twelve_cases_cover_required_inputs_and_remain_pending(self):
        cases = self.manifest['cases']
        self.assertEqual(len(cases), 12)
        self.assertTrue(any('中文' in case['prompt'] for case in cases))
        self.assertTrue(any('駅' in case['prompt'] for case in cases))
        self.assertTrue(any('\n' in case['prompt'] for case in cases))
        self.assertTrue(any(case['prompt'] != case['prompt'].strip() for case in cases))
        self.assertTrue(any(case['max_tokens'] == 2048 for case in cases))
        near = next(case for case in cases if case['id'] == 'pe-dev-11-near-capacity')
        self.assertGreaterEqual(near['input_tokens'], 2000)
        self.assertLessEqual(near['input_tokens'], 2048)
        self.assertEqual(self.manifest['status'], 'inputs_frozen_model_acceptance_pending')
        self.assertTrue(all(case['acceptance_status'] == 'pending_real_official_and_native_pe' for case in cases))

    def test_contract_rejects_changed_prompt_hash_and_implicit_sampling(self):
        changed = copy.deepcopy(self.manifest)
        changed['cases'][0]['prompt'] += ' '
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_development_batch(changed, self.tokenizer)
        changed = copy.deepcopy(self.manifest)
        del changed['cases'][0]['top_p']
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_development_batch(changed, self.tokenizer)

    def test_contract_rejects_over_capacity_without_model_execution(self):
        changed = copy.deepcopy(self.manifest)
        case = changed['cases'][-1]
        case['prompt'] = 'cat ' * 3000
        unsigned = dict(changed)
        unsigned.pop('manifest_sha256')
        import hashlib
        changed['manifest_sha256'] = hashlib.sha256(json.dumps(
            unsigned, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'identity'):
            validate_development_batch(changed, self.tokenizer)

    def test_frozen_contract_matches_pinned_tokenizer(self):
        root = Path(__file__).parents[1]
        manifest = json.loads((root / 'tests/fixtures/pe-development-batch.json').read_text())
        tokenizer = AutoTokenizer.from_pretrained(root / 'models/pe-tokenizer', local_files_only=True)
        validate_development_batch(manifest, tokenizer)
        self.assertEqual(manifest['cases'][-1]['input_tokens'], 2048)


if __name__ == '__main__':
    unittest.main()
