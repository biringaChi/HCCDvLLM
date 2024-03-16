import torch
from dataset import Dataset
from sklearn.model_selection import train_test_split
from simpletransformers.config.model_args import ModelArgs
from simpletransformers.language_representation import RepresentationModel

class Repr(Dataset):
	def __init__(self) -> None:
		super().__init__()

	def _batch(self, sequence, steps = 1):
		for idx in range(0, len(sequence), steps):
			yield sequence[idx : min(idx + steps, len(sequence))]
	
	def splits(self, x):
		X_train, Xs, = train_test_split(x, train_size = 0.8, random_state = 1)
		X_val, X_test = train_test_split(Xs, test_size = 0.5, random_state = 1)
		return X_train, X_val, X_test

	def _repr_model(self, sequence, model_type, model_name, batch_n = 32):
		use_cuda = True if torch.cuda.is_available() else False
		model_args = ModelArgs(num_train_epochs = 4)
		hidden_states = []
		model = RepresentationModel(model_type = model_type, model_name = model_name, args = model_args, use_cuda = use_cuda)
		for x in self._batch(sequence, batch_n):
			hidden_states.append(model.encode_sentences(x, combine_strategy = "mean", batch_size = len(x)))
		return [i for vector in hidden_states for i in vector]
	
	def _stream_splits(self):
		password_train, password_val, password_test = self.splits(self._get_passwords())
		generic_secret_train, generic_secret_val, generic_secret_test = self.splits(self._get_generic_secrets())
		private_key_train, private_key_val, private_key_test = self.splits(self._get_private_keys())
		generic_token_train, generic_token_val, generic_token_test = self.splits(self._get_generic_tokens())
		predefined_pattern_train, predefined_pattern_val, predefined_pattern_test = self.splits(self._get_predefined_patterns())
		auth_key_token_train, auth_key_token_val, auth_key_token_test = self.splits(self._get_auth_key_tokens())
		seed_salt_nonce_train, seed_salt_nonce_val, seed_salt_nonce_test = self.splits(self._get_seed_salt_nonces())
		other_train, other_val, other_test = self.splits(self._get_others())

		return {
			"passwords" : (password_train, password_val, password_test),
			"generic_secrets" : (generic_secret_train, generic_secret_val, generic_secret_test),
			"private_keys" : (private_key_train, private_key_val, private_key_test),
			"generic_tokens" : (generic_token_train, generic_token_val, generic_token_test),
			"predefined_patterns": (predefined_pattern_train, predefined_pattern_val, predefined_pattern_test),
			"auth_key_tokens": (auth_key_token_train, auth_key_token_val, auth_key_token_test),
			"seed_salt_nonces" : (seed_salt_nonce_train, seed_salt_nonce_val, seed_salt_nonce_test),
			"others" : (other_train, other_val, other_test)
		}
	
	def _stream_features(self, model_type, model_name):
		passwords_train = self._repr_model(self._stream_splits()["passwords"][0], model_type, model_name)
		passwords_val = self._repr_model(self._stream_splits()["passwords"][1], model_type, model_name)
		passwords_test = self._repr_model(self._stream_splits()["passwords"][2], model_type, model_name)
		passwords = passwords_train + passwords_val + passwords_test

		generic_secrets_train = self._repr_model(self._stream_splits()["generic_secrets"][0], model_type, model_name)
		generic_secrets_val = self._repr_model(self._stream_splits()["generic_secrets"][1], model_type, model_name)
		generic_secrets_test = self._repr_model(self._stream_splits()["generic_secrets"][2], model_type, model_name)
		generic_secrets = generic_secrets_train + generic_secrets_val + generic_secrets_test

		private_keys_train = self._repr_model(self._stream_splits()["private_keys"][0], model_type, model_name)
		private_keys_val = self._repr_model(self._stream_splits()["private_keys"][1], model_type, model_name)
		private_keys_test = self._repr_model(self._stream_splits()["private_keys"][2], model_type, model_name)
		private_keys = private_keys_train + private_keys_val + private_keys_test

		generic_tokens_train = self._repr_model(self._stream_splits()["generic_tokens"][0], model_type, model_name)
		generic_tokens_val = self._repr_model(self._stream_splits()["generic_tokens"][1], model_type, model_name)
		generic_tokens_test = self._repr_model(self._stream_splits()["generic_tokens"][2], model_type, model_name)
		generic_tokens = generic_tokens_train + generic_tokens_val + generic_tokens_test

		predefined_patterns_train = self._repr_model(self._stream_splits()["predefined_patterns"][0], model_type, model_name)
		predefined_patterns_val = self._repr_model(self._stream_splits()["predefined_patterns"][1], model_type, model_name)
		predefined_patterns_test = self._repr_model(self._stream_splits()["predefined_patterns"][2], model_type, model_name)
		predefined_patterns = predefined_patterns_train + predefined_patterns_val + predefined_patterns_test

		auth_key_tokens_train = self._repr_model(self._stream_splits()["auth_key_tokens"][0], model_type, model_name)
		auth_key_tokens_val = self._repr_model(self._stream_splits()["auth_key_tokens"][1], model_type, model_name)
		auth_key_tokens_test = self._repr_model(self._stream_splits()["auth_key_tokens"][2], model_type, model_name)
		auth_key_tokens = auth_key_tokens_train + auth_key_tokens_val + auth_key_tokens_test

		seed_salt_nonces_train = self._repr_model(self._stream_splits()["seed_salt_nonces"][0], model_type, model_name)
		seed_salt_nonces_val = self._repr_model(self._stream_splits()["seed_salt_nonces"][1], model_type, model_name)
		seed_salt_nonces_test = self._repr_model(self._stream_splits()["seed_salt_nonces"][2], model_type, model_name)
		seed_salt_nonces = seed_salt_nonces_train + seed_salt_nonces_val + seed_salt_nonces_test

		others_train = self._repr_model(self._stream_splits()["others"][0], model_type, model_name)
		others_val = self._repr_model(self._stream_splits()["others"][1], model_type, model_name)
		others_test = self._repr_model(self._stream_splits()["others"][2], model_type, model_name)
		others = others_train + others_val + others_test

		return passwords + generic_secrets + private_keys + generic_tokens + predefined_patterns + auth_key_tokens + seed_salt_nonces + others
	