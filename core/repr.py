from dataset import Dataset
from simpletransformers.config.model_args import ModelArgs
from simpletransformers.language_representation import RepresentationModel

class Repr(Dataset):
	def __init__(self) -> None:
		super().__init__()

	def _batch(self, sequence, steps = 1):
		for idx in range(0, len(sequence), steps):
			yield sequence[idx : min(idx + steps, len(sequence))]

	def _repr_model(self, sequence, model_type, model_name, use_cuda, batch_n = 32):
		model_args = ModelArgs(num_train_epochs = 4)
		vectors = []
		model = RepresentationModel(model_type = model_type, model_name = model_name, args = model_args, use_cuda = use_cuda)
		for x in self._batch(sequence, batch_n):
			vectors.append(model.encode_sentences(x, combine_strategy = "mean", batch_size = len(x)))
		return [i for vector in vectors for i in vector]
	
	def bert_repr(self):
		return self._repr_model(self.get_data(), "bert", "bert-base-uncased", True) 
	
	def gpt_repr(self):
		return self._repr_model(self.get_data(), "gpt2", "gpt2", True)