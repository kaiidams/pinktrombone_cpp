from pinktrombone import PinkTrombone
import numpy as np
import soundfile as sf


with open('dump.bin', 'rb') as fp:
    x = np.frombuffer(fp.read(), dtype=np.float32)
    x = x.reshape(-1, 50)
    assert np.all(x[:, 0] == 50.0)
    data = x[:, 1:].astype(np.float64)

pt = PinkTrombone(44)
out = []
for i in range(data.shape[0]):
    x = data[i].tobytes()
    pt.control(x)
    y = pt.process()
    out.append(y)
out = np.concatenate(out)
print(out.shape)
sf.write("output.wav", out, samplerate=44100)
