import pytest
from voice100_pinktrombone import PinkTrombone
import numpy as np
import soundfile as sf


def convert():
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


def dummy_data(N=44, T=2000, M=100):
    import numpy as np

    X = np.random.normal(size=(T, N + 5))
    w = np.hamming(M)
    w = w / np.sum(w)
    Y = X[M - 1:, :] * w[0]
    for i in range(1, M):
        # print(X[M-1-i:-i, :].shape)
        Y = Y + X[M-1-i:-i, :] * w[i]
    X = Y
    del Y

    X[:, 0] = X[:, 0] > -.1
    X[:, 1] = np.clip(X[:, 1] * 30 + 140 + (44 - N) * 8, 60, 250)
    X[:, 2:4] = np.clip(X[:, 2:4] * 5 + 0.7, 0.0, 1.0)
    X[:, 3] = X[:, 2] ** 0.8
    X[:, 4] = np.clip(X[:, 4] * 2 + 0.1, 0.0, 0.2)
    X[:, 6:-1] = (X[:, 5:-2] + X[:, 6:-1] + X[:, 7:]) * 2 + 3.0
    X[:, 5:] = np.clip(X[:, 5:], -0.1, 5.0)

    return X


@pytest.mark.parametrize("N", [44, 35])
def test_convert(N):
    X = dummy_data(N)
    pt = PinkTrombone(N)
    Y = []
    for i in range(X.shape[0]):
        pt.control(X[i, :])
        y = pt.process()
        Y.append(y)
    Y = np.concatenate(Y)
    assert Y.shape[0] == X.shape[0] * 512
