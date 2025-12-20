# Экспорт в TensorFlow Lite для embedded (ARM в капсуле)
import tensorflow as tf
from torch.onnx import export

# ONNX -> TFLite
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(model, dummy_input, 'model.onnx')
# Затем конвертируйте ONNX в TFLite с помощью tf2onnx
