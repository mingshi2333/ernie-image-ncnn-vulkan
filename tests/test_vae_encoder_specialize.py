import unittest
from tools.export_vae_encoder import dimensions
from tools.specialize_vae_encoder import reviewed_dimensions

class EncoderLargeShapeContract(unittest.TestCase):
 def test_large_shape_requires_explicit_fixture_mode(self):
  with self.assertRaises(ValueError):dimensions(512,384)
  reviewed_dimensions(512,384)
 def test_only_reviewed_large_shape_is_admitted(self):
  for shape in [(384,512),(512,512),(256,384),(513,384)]:
   with self.subTest(shape=shape),self.assertRaises(ValueError):reviewed_dimensions(*shape)
 def test_small_contract_unchanged(self):
  dimensions(32,32);dimensions(64,32)
  for shape in [(15,16),(80,32),(48,18)]:
   with self.subTest(shape=shape),self.assertRaises(ValueError):dimensions(*shape)

if __name__=='__main__':unittest.main()
