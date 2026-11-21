"""Time an ONNX model on CPU. Numbers from this script are CPU, not Metis."""
import os, sys, time
import numpy as np
import onnxruntime as ort

path = sys.argv[1]
shape = [int(x) for x in sys.argv[2].split(",")]
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name
x = np.random.rand(*shape).astype(np.float32)

for _ in range(10):          # warm-up
    sess.run(None, {name: x})
times = []
for _ in range(100):
    start = time.perf_counter()
    sess.run(None, {name: x})
    times.append(time.perf_counter() - start)

median = float(np.median(times))
print(f"{os.path.basename(path)}: median {median*1000:.1f} ms, {1/median:.1f} FPS, "
      f"size {os.path.getsize(path)/1e6:.1f} MB (CPU, not Metis)")
