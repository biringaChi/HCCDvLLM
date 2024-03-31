import pickle

class Dataset:
	def __init__(self) -> None:
		self.hcc = "hcc"
		
	def unpickle(self, data):
		with open(data, "rb") as file:
			loaded = pickle.load(file)
		return loaded
	
	def _get_passwords(self):
		return self.unpickle(f"{self.hcc}/password.pkl")
	
	def _get_passwords_labs(self):
		return [0 for _ in range(len(self._get_passwords()))]

	def _get_generic_secrets(self):
		return self.unpickle(f"{self.hcc}/generic_secret.pkl")
	
	def _get_generic_secrets_labs(self):
		return [1 for _ in range(len(self._get_generic_secrets()))]
	
	def _get_private_keys(self):
		return self.unpickle(f"{self.hcc}/private_key.pkl")
	
	def _get_private_keys_labs(self):
		return [2 for _ in range(len(self._get_private_keys()))]
	
	def _get_generic_tokens(self):
		return self.unpickle(f"{self.hcc}/generic_token.pkl")

	def _get_generic_tokens_labs(self):
		return [3 for _ in range(len(self._get_generic_tokens()))]

	def _get_predefined_patterns(self):
		return self.unpickle(f"{self.hcc}/predefined_pattern.pkl")

	def _get_predefined_patterns_labs(self):
		return [4 for _ in range(len(self._get_predefined_patterns()))]
	
	def _get_auth_key_tokens(self):
		return self.unpickle(f"{self.hcc}/auth_key_token.pkl")

	def _get_auth_key_tokens_labs(self):
		return [5 for _ in range(len(self._get_auth_key_tokens()))]

	def _get_seed_salt_nonces(self):
		return self.unpickle(f"{self.hcc}/seed_salt_nonce.pkl")

	def _get_seed_salt_nonces_labs(self):
		return [6 for _ in range(len(self._get_seed_salt_nonces()))]

	def _get_others(self):
		return self.unpickle(f"{self.hcc}/other.pkl")
	
	def _get_others_labs(self):
		return [7 for _ in range(len(self._get_others()))]
	
	def _get_data(self):
		return self._get_passwords() + self._get_generic_secrets() + self._get_private_keys() + self._get_generic_tokens() + \
		self._get_predefined_patterns() + self._get_auth_key_tokens() + self._get_seed_salt_nonces() + self._get_others()
	
	def _get_labels(self):
		return self._get_passwords_labs() + self._get_generic_secrets_labs() + self._get_private_keys_labs() + self._get_generic_tokens_labs() + \
		self._get_predefined_patterns_labs() + self._get_auth_key_tokens_labs() + self._get_seed_salt_nonces_labs() + self._get_others_labs()
	