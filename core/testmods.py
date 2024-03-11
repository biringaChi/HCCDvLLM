import unittest
from dataset import Dataset

class Tests(unittest.TestCase):
	def setUp(self) -> None:
		self.dataset = Dataset()
	
	def test_passwords(self):
		self.assertEqual(len(self.dataset._get_passwords()), 1395)
		self.assertTrue(map(lambda x: isinstance(x, str), self.dataset._get_passwords()))
	
	def test_generic_secrets(self):
		self.assertEqual(len(self.dataset._get_generic_secrets()), 1056)
		self.assertTrue(map(lambda x: isinstance(x, str), self.dataset._get_generic_secrets()))
	
	def test_private_keys(self):
		self.assertEqual(len(self.dataset._get_private_keys()), 992)
		self.assertTrue(map(lambda x: isinstance(x, str), self.dataset._get_private_keys()))

	def test_generic_tokens(self):
		self.assertEqual(len(self.dataset._get_generic_tokens()), 333)
		self.assertTrue(map(lambda x: isinstance(x, str), self.dataset._get_generic_tokens()))

	def test_predefined_patterns(self):
		self.assertEqual(len(self.dataset._get_predefined_patterns()), 326)
		self.assertTrue(map(lambda x: isinstance(x, str), self.dataset._get_predefined_patterns()))

	def test_auth_key_tokens(self):
		self.assertEqual(len(self.dataset._get_auth_key_tokens()), 67)
		self.assertTrue(map(lambda x: isinstance(x, str), self.dataset._get_auth_key_tokens()))

	def test_seed_salt_nonces(self):
		self.assertEqual(len(self.dataset._get_seed_salt_nonces()), 39)
		self.assertTrue(map(lambda x: isinstance(x, str), self.dataset._get_seed_salt_nonces()))

	def test_others(self):
		self.assertEqual(len(self.dataset._get_others()), 374)
		self.assertTrue(map(lambda x: isinstance(x, str), self.dataset._get_others()))

	def test_passwords_labs(self):
		self.assertTrue(map(lambda x: x == 0, self.dataset._get_passwords_labs()))
		self.assertTrue(map(lambda x: isinstance(x, int), self.dataset._get_passwords_labs()))

	def test_generic_secrets_labs(self):
		self.assertTrue(map(lambda x: x == 1, self.dataset._get_generic_secrets_labs()))
		self.assertTrue(map(lambda x: isinstance(x, int), self.dataset._get_generic_secrets_labs()))

	def test_private_keys_labs(self):
		self.assertTrue(map(lambda x: x == 2, self.dataset._get_private_keys_labs()))
		self.assertTrue(map(lambda x: isinstance(x, int), self.dataset._get_private_keys_labs()))

	def test_generic_tokens_labs(self):
		self.assertTrue(map(lambda x: x == 3, self.dataset._get_generic_tokens_labs()))
		self.assertTrue(map(lambda x: isinstance(x, int), self.dataset._get_generic_tokens_labs()))

	def test_predefined_patterns_labs(self):
		self.assertTrue(map(lambda x: x == 4, self.dataset._get_predefined_patterns_labs()))
		self.assertTrue(map(lambda x: isinstance(x, int), self.dataset._get_predefined_patterns_labs()))

	def test_auth_key_tokens_labs(self):
		self.assertTrue(map(lambda x: x == 5, self.dataset._get_auth_key_tokens_labs()))
		self.assertTrue(map(lambda x: isinstance(x, int), self.dataset._get_auth_key_tokens_labs()))

	def test_seed_salt_nonces(self):
		self.assertTrue(map(lambda x: x == 6, self.dataset._get_seed_salt_nonces_labs()))
		self.assertTrue(map(lambda x: isinstance(x, int), self.dataset._get_seed_salt_nonces_labs()))

	def test_others(self):
		self.assertTrue(map(lambda x: x == 7, self.dataset._get_others_labs()))
		self.assertTrue(map(lambda x: isinstance(x, int), self.dataset._get_others_labs()))

if __name__ == "__main__":
	unittest.main()