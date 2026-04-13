import numpy as np

from train_final_model import choose_final_epochs, make_seq2seq_training_arrays


def test_choose_final_epochs_prefers_largest_signal():
    assert choose_final_epochs(8, 14, 10) == 14
    assert choose_final_epochs(18, 14, 10) == 18
    assert choose_final_epochs(6, None, 10) == 10


def test_make_seq2seq_training_arrays_shapes():
    x_scaled = np.arange(60, dtype=np.float32).reshape(12, 5)
    y_scaled = np.arange(12, dtype=np.float32)

    x_seq, decoder_seq, y_seq = make_seq2seq_training_arrays(
        x_scaled,
        y_scaled,
        lookback=4,
        horizon=1,
    )

    assert x_seq.shape == (8, 4, 5)
    assert decoder_seq.shape == (8, 1, 1)
    assert y_seq.shape == (8, 1)
    assert decoder_seq[0, 0, 0] == y_scaled[3]
    assert y_seq[0, 0] == y_scaled[4]
