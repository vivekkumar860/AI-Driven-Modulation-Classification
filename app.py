import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.graph_objs as go
import io
import os

# Modulation classes
MOD_CLASSES = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'AM-DSB', 'AM-SSB', 'WBFM', 'GFSK', 'PAM4', 'CPFSK']

# Sidebar
st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/1/17/Google-flutter-logo.png', width=80)  # Placeholder logo
st.sidebar.title('Modulation Classifier')
st.sidebar.markdown('''
**Model:** Trained Keras model (`mod_classifier.h5`)

**Input:** NumPy `.npy` file with I/Q samples, shape (128, 2) or (N, 128, 2)

**Output:** Softmax probabilities over 11 modulation classes

**Usage:**
- Upload a `.npy` file containing I/Q samples.
- The model will classify the modulation type.

**Contact:**
- Email: your.email@domain.com
- GitHub: [your-github](https://github.com/your-github)

**Resources:**
- [Streamlit Docs](https://docs.streamlit.io/)
- [TensorFlow Docs](https://www.tensorflow.org/)
''')

st.title('Modulation Classification Demo')
st.write('Upload a NumPy `.npy` file containing I/Q samples (shape: 128x2 or N x 128 x 2) to classify the modulation type.')

# Example file for download
def get_example_iq():
    np.random.seed(42)
    arr = np.random.randn(128, 2).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return buf

st.download_button(
    label="Download Example I/Q .npy File",
    data=get_example_iq(),
    file_name="example_iq.npy",
    mime="application/octet-stream"
)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('mod_classifier.h5')
        return model
    except Exception:
        return None

model = load_model()

if model is None:
    st.error("Model file 'mod_classifier.h5' not found or could not be loaded. Please ensure the file is present in the app directory.")
    st.stop()

# Model summary
with st.expander("Show Model Summary"):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    summary = '\n'.join(stringlist)
    st.code(summary)

# Reset button
if st.button('Reset App'):
    st.experimental_rerun()

uploaded_file = st.file_uploader('Upload I/Q samples (.npy file)', type=['npy'])

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        iq_data = np.load(uploaded_file, allow_pickle=False)
    except Exception as e:
        st.error(f"Could not read .npy file: {e}")
    else:
        # Accept (128,2) or (N,128,2)
        if not isinstance(iq_data, np.ndarray):
            st.error("Uploaded file is not a valid NumPy array.")
        elif iq_data.shape == (128, 2):
            iq_data = iq_data.astype(np.float32)
            iq_data_exp = np.expand_dims(iq_data, axis=0)
        elif len(iq_data.shape) == 3 and iq_data.shape[1:] == (128, 2):
            iq_data = iq_data.astype(np.float32)
            iq_data_exp = iq_data
        else:
            st.error(f"Invalid input shape: {iq_data.shape}. Expected (128, 2) or (N, 128, 2).")
            st.stop()
        # Show file info
        st.info(f"File shape: {iq_data.shape}, dtype: {iq_data.dtype}")
        # Show stats
        st.write('**Input Data Statistics:**')
        stats = pd.DataFrame({
            'Channel': ['I', 'Q'],
            'Mean': np.mean(iq_data[..., :2], axis=(-2, 0)).tolist() if iq_data.ndim == 3 else np.mean(iq_data, axis=0).tolist(),
            'Std': np.std(iq_data[..., :2], axis=(-2, 0)).tolist() if iq_data.ndim == 3 else np.std(iq_data, axis=0).tolist(),
            'Min': np.min(iq_data[..., :2], axis=(-2, 0)).tolist() if iq_data.ndim == 3 else np.min(iq_data, axis=0).tolist(),
            'Max': np.max(iq_data[..., :2], axis=(-2, 0)).tolist() if iq_data.ndim == 3 else np.max(iq_data, axis=0).tolist(),
        })
        st.dataframe(stats, hide_index=True)
        # Prediction
        with st.spinner('Running prediction...'):
            try:
                preds = model.predict(iq_data_exp)
                # If batch, show for first sample and allow selection
                if preds.shape[0] > 1:
                    idx = st.number_input('Select sample in batch', min_value=0, max_value=preds.shape[0]-1, value=0, step=1)
                else:
                    idx = 0
                pred_idx = int(np.argmax(preds[idx]))
                pred_class = MOD_CLASSES[pred_idx]
                confidence = float(preds[idx][pred_idx])
                # Metrics display
                col1, col2 = st.columns(2)
                col1.metric("Prediction", pred_class)
                col2.metric("Confidence", f"{confidence:.2%}")
                # Probability bar chart
                st.write('### Class Probabilities')
                prob_df = pd.DataFrame({
                    'Modulation': MOD_CLASSES,
                    'Probability': preds[idx]
                })
                prob_df = prob_df.sort_values('Probability', ascending=False).reset_index(drop=True)
                fig_bar = go.Figure([go.Bar(x=prob_df['Modulation'], y=prob_df['Probability'])])
                fig_bar.update_layout(yaxis=dict(tickformat='.0%'), title='Class Probability Distribution')
                st.plotly_chart(fig_bar, use_container_width=True)
                # Interactive I/Q plot
                st.write('### I/Q Waveforms')
                iq_plot = go.Figure()
                iq_plot.add_trace(go.Scatter(y=iq_data_exp[idx, :, 0], mode='lines', name='I (In-phase)'))
                iq_plot.add_trace(go.Scatter(y=iq_data_exp[idx, :, 1], mode='lines', name='Q (Quadrature)'))
                iq_plot.update_layout(title='I/Q Waveforms', xaxis_title='Sample Index', yaxis_title='Amplitude')
                st.plotly_chart(iq_plot, use_container_width=True)
            except Exception as e:
                st.error(f"Error during prediction or plotting: {e}")
else:
    st.info('Awaiting file upload.') 