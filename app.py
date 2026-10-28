from pathlib import Path
import tempfile

import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import torch
from src.data import features, read_audio
from src.model import SoundCNN

st.title('Что это за звук?')
st.write('Загрузите WAV-файл: модель классифицирует первые пять секунд среди 10 классов ESC-10.')
checkpoint_path = Path('checkpoints/best.pt')
if not checkpoint_path.exists():
    st.info('Сначала обучите модель командой из README: checkpoint пока отсутствует.')
    st.stop()
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
model = SoundCNN(len(checkpoint['labels']))
model.load_state_dict(checkpoint['state_dict'])
model.eval()
uploaded = st.file_uploader('Аудиофайл WAV', type=['wav'])
if uploaded:
    st.audio(uploaded)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'audio.wav'
        path.write_bytes(uploaded.getvalue())
        try:
            audio = read_audio(path)
            spec = features(audio)
        except Exception as exc:
            st.error(f'Не удалось прочитать аудио: {exc}')
            st.stop()
    fig, ax = plt.subplots()
    librosa.display.specshow(spec.squeeze().numpy(), x_axis='time', y_axis='mel', sr=22050, hop_length=512, ax=ax)
    ax.set_title('Нормализованная log-mel спектрограмма')
    st.pyplot(fig)
    with torch.no_grad():
        probs = model(spec.unsqueeze(0)).softmax(dim=1)[0]
    for i in probs.argsort(descending=True)[:3]:
        st.write(f"{checkpoint['labels'][int(i)]}: {probs[i].item():.1%}")
    st.caption('Оценки модели не обязательно откалиброваны; незнакомые звуки она тоже отнесёт к одному из 10 классов.')
