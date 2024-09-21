from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


def build_lstm_model(actions):
    """Create and compile an LSTM model."""
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def train_model(model, X_train, y_train, log_dir="Logs"):
    """Train the LSTM model and save logs."""
    tb_callback = TensorBoard(log_dir=log_dir)
    model.fit(X_train, y_train, epochs=4000, callbacks=[tb_callback])


def save_model(model, filename='action.h5'):
    """Save the trained model to disk."""
    model.save(filename)


def load_model(filename='action.h5'):
    """Load a pre-trained model from disk."""
    from tensorflow.keras.models import load_model
    return load_model(filename)
