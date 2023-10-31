import onnxruntime 
import numpy as np 
 
sess = onnxruntime.InferenceSession('D:\project\some_onnx\linear_func.onnx') 
a = np.random.rand(10, 10).astype(np.float32) 
b = np.random.rand(10, 10).astype(np.float32) 
x = np.random.rand(10, 10).astype(np.float32) 
 
output = sess.run(['output'], {'a': a, 'b': b, 'x': x})[0] 
 
assert np.allclose(output, a * x + b)