"""Reject upstream drift before deriving production mask shaders."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('derive_sdpa_broadcast', ROOT / 'cmake/derive_sdpa_broadcast.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BroadcastSourceTests(unittest.TestCase):
    def test_pinned_variants_and_crlf(self):
        for name, sha in module.SHADERS.items():
            path = ROOT / 'third_party/ncnn/src/layer/vulkan/shader' / (name + '.comp')
            source = module.authenticated(path, sha)
            self.assertIn('int mask_h;', module.shader(name, source))
            with tempfile.TemporaryDirectory() as tmp:
                windows = Path(tmp) / 'source.comp'
                windows.write_bytes(source.replace('\n', '\r\n').encode())
                self.assertEqual(module.authenticated(windows, sha), source)
                windows.write_text(source + '// upstream changed\n')
                with self.assertRaisesRegex(ValueError, 'Review changed ncnn source'):
                    module.authenticated(windows, sha)

    def test_incomplete_transformation_is_rejected(self):
        for name in module.SHADERS:
            with self.assertRaises(ValueError):
                module.shader(name, 'void main() {}')


if __name__ == '__main__':
    unittest.main()
