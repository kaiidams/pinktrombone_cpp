from pinktrombone import PinkTrombone
import numpy as np
import soundfile as sf

N = 44

with open('dump.bin', 'rb') as fp:
    x = np.frombuffer(fp.read(), dtype=np.float32)
    x = x.reshape(-1, N + 6)
    assert np.all(x[:, 0] == N + 6)
    data = x[:, 1:].astype(np.float64)

pt = PinkTrombone(N)
if True:
    out = pt(data)
    assert out.shape[0] == data.shape[0] * 512
else:
    out = []
    for i in range(data.shape[0]):
        x = data[i, :N]
        pt.control(x)
        y = pt.process()
        out.append(y)
    out = np.concatenate(out)
print(out.shape)
sf.write("output.wav", out, samplerate=44100)
