import torch

from cnn_page_classifier import CNNPageClassifier
from text_to_tensor import text_to_tensor


model = CNNPageClassifier()

sample_text = "Ruling 108. Wine is impure..."

tensor = text_to_tensor(sample_text)

output = model(tensor)

print("CNN output:", output)