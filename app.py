import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import onnxruntime as ort
import io
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
import base64

# --- Custom CSS for header, sidebar, dark mode, favicon ---
st.set_page_config(page_title="Modulation Classifier", page_icon="📡", layout="wide")

st.markdown("""
    <style>
    .main-header {
        font-size:2.5rem;
        color:#4F8BF9;
        font-weight:bold;
        margin-bottom:0.5em;
    }
    .about-box {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 1em;
        margin-bottom: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# --- Dark mode toggle ---
dark_mode = st.sidebar.toggle("🌙 Dark mode", value=False)
if dark_mode:
    st.markdown("""
        <style>
        body, .main, .stApp { background-color: #18191A !important; color: #F0F2F6 !important; }
        .main-header { color: #F0F2F6 !important; }
        .about-box { background-color: #23272F !important; color: #F0F2F6 !important; }
        </style>
    """, unsafe_allow_html=True)

# --- Sidebar with collapsible sections ---
st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/1/17/Google-flutter-logo.png', width=80)
with st.sidebar.expander("About", expanded=True):
    st.markdown('<div class="about-box"><b>About:</b><br>This app classifies radio modulation types from I/Q samples using a deep learning model. Upload your own model or use the default!</div>', unsafe_allow_html=True)
with st.sidebar.expander("Contact/Links", expanded=False):
    st.markdown('''
    **Contact:**  
    Email: your.real.email@domain.com  
    GitHub: [your-github](https://github.com/your-github)
    ''')
    st.markdown('''
    **Resources:**  
    - [Streamlit Docs](https://docs.streamlit.io/)  
    - [TensorFlow Docs](https://www.tensorflow.org/)
    ''')

# --- Main Header ---
st.markdown('<div class="main-header">Modulation Classification Demo</div>', unsafe_allow_html=True)

# --- Tabs for main content ---
tabs = st.tabs(["App", "Help/Docs"])

with tabs[1]:
    st.markdown("""
    ### How to Use
    1. Upload one or more `.npy` files (single or batch).
    2. Optionally upload your own Keras `.h5` model in the sidebar.
    3. Explore the results in the organized tabs.
    4. Download probabilities and prediction history as CSV.
    5. View constellation diagrams and data stats.
    6. Enjoy a modern, interactive experience!
    
    **Advanced Features:**
    - Confusion matrix (if you upload ground truth labels)
    - SNR estimation (if your data allows)
    - Download constellation diagram as image
    - Dark mode toggle
    - REST API endpoint (see docs)
    - Webhook/email notification (stub)
    """)

with tabs[0]:
    st.write('Upload a NumPy `.npy` file containing I/Q samples (shape: 128x2 or N x 128 x 2) to classify the modulation type.')

    # --- Example file for download ---
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

    # --- Allow user to upload their own model ---
    def save_uploaded_model(uploaded_model):
        with open('user_model.h5', 'wb') as f:
            f.write(uploaded_model.read())
        return 'user_model.h5'

    user_model_file = st.sidebar.file_uploader('Upload your Keras model (.h5)', type=['h5'], help='Optional: Use your own trained Keras model.')

    # --- ONNX Model Loading ---
    @st.cache_resource
    def load_onnx_model():
        try:
            session = ort.InferenceSession("mod_classifier.onnx")
            return session
        except Exception as e:
            st.error(f"Error loading ONNX model: {e}")
            return None

    session = load_onnx_model()
    if session is None:
        st.error("ONNX model file 'mod_classifier.onnx' not found or could not be loaded. Please ensure the file is present in the app directory.")
        st.stop()

    # --- Prediction history (session state) ---
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'constellations' not in st.session_state:
        st.session_state['constellations'] = []

    # --- File uploader (drag-and-drop, batch support) ---
    uploaded_files = st.file_uploader('Upload I/Q samples (.npy file, single or batch)', type=['npy'], accept_multiple_files=True, help='You can drag and drop multiple files for batch processing.')
    gt_labels_file = st.file_uploader('Upload ground truth labels (.npy, optional)', type=['npy'], help='Optional: For confusion matrix, upload a .npy file of integer labels.')

    MOD_CLASSES = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'AM-DSB', 'AM-SSB', 'WBFM', 'GFSK', 'PAM4', 'CPFSK']

    # --- Helper for constellation diagram ---
    def plot_constellation(iq_data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=iq_data[:, 0], y=iq_data[:, 1], mode='markers', marker=dict(size=6, color='#4F8BF9'), name='Constellation'))
        fig.update_layout(title='Constellation Diagram (I vs Q)', xaxis_title='I (In-phase)', yaxis_title='Q (Quadrature)', width=500, height=400)
        return fig

    def get_constellation_image(fig):
        img_bytes = fig.to_image(format="png")
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="constellation.png">Download Constellation Diagram</a>'
        return href

    # --- Helper for CSV download ---
    def get_csv_download_link(df, filename='results.csv'):
        return df.to_csv(index=False).encode('utf-8')

    # --- Helper for SNR estimation ---
    def estimate_snr(iq):
        # Simple SNR estimation: mean(signal^2) / mean(noise^2)
        # Here, treat signal as mean-centered, noise as deviation
        signal_power = np.mean(iq ** 2)
        noise_power = np.var(iq)
        if noise_power == 0:
            return np.nan
        return 10 * np.log10(signal_power / noise_power)

    # --- Prediction function using ONNX ---
    def predict_onnx(iq_data):
        # iq_data shape: (batch, 128, 2)
        input_name = session.get_inputs()[0].name
        inputs = {input_name: iq_data.astype(np.float32)}
        preds = session.run(None, inputs)[0]
        return preds

    # --- Main logic ---
    all_preds = []
    all_true = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                uploaded_file.seek(0)
                iq_data = np.load(uploaded_file, allow_pickle=False)
            except Exception as e:
                st.error(f"Could not read .npy file: {e}")
                continue
            # Accept (128,2) or (N,128,2)
            if not isinstance(iq_data, np.ndarray):
                st.error("Uploaded file is not a valid NumPy array.")
                continue
            elif iq_data.shape == (128, 2):
                iq_data = iq_data.astype(np.float32)
                iq_data_exp = np.expand_dims(iq_data, axis=0)
            elif len(iq_data.shape) == 3 and iq_data.shape[1:] == (128, 2):
                iq_data = iq_data.astype(np.float32)
                iq_data_exp = iq_data
            else:
                st.error(f"Invalid input shape: {iq_data.shape}. Expected (128, 2) or (N, 128, 2).")
                continue
            st.toast(f"File '{uploaded_file.name}' loaded successfully!", icon='✅')
            # Show file info
            st.info(f"File: {uploaded_file.name} | Shape: {iq_data.shape} | dtype: {iq_data.dtype}")
            # Show stats
            stats = pd.DataFrame({
                'Channel': ['I', 'Q'],
                'Mean': np.mean(iq_data[..., :2], axis=(-2, 0)).tolist() if iq_data.ndim == 3 else np.mean(iq_data, axis=0).tolist(),
                'Std': np.std(iq_data[..., :2], axis=(-2, 0)).tolist() if iq_data.ndim == 3 else np.std(iq_data, axis=0).tolist(),
                'Min': np.min(iq_data[..., :2], axis=(-2, 0)).tolist() if iq_data.ndim == 3 else np.min(iq_data, axis=0).tolist(),
                'Max': np.max(iq_data[..., :2], axis=(-2, 0)).tolist() if iq_data.ndim == 3 else np.max(iq_data, axis=0).tolist(),
            })
            # --- Prediction ---
            with st.spinner('Running prediction...'):
                try:
                    preds = predict_onnx(iq_data_exp)
                    # If batch, show for first sample and allow selection
                    if preds.shape[0] > 1:
                        idx = st.number_input('Select sample in batch', min_value=0, max_value=preds.shape[0]-1, value=0, step=1, key=f'batch_{uploaded_file.name}')
                    else:
                        idx = 0
                    pred_idx = int(np.argmax(preds[idx]))
                    pred_class = MOD_CLASSES[pred_idx]
                    confidence = float(preds[idx][pred_idx])
                    # Save to history
                    st.session_state['history'].append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'file': uploaded_file.name,
                        'prediction': pred_class,
                        'confidence': confidence
                    })
                    all_preds.append(pred_idx)
                    # --- Tabs for output organization ---
                    subtabs = st.tabs(["Prediction", "Probabilities", "Waveforms", "Constellation", "Data Stats", "Model Info", "SNR"])
                    with subtabs[0]:
                        st.success(f"**Prediction:** {pred_class}")
                        st.metric("Confidence", f"{confidence:.2%}")
                    with subtabs[1]:
                        prob_df = pd.DataFrame({
                            'Modulation': MOD_CLASSES,
                            'Probability': preds[idx]
                        })
                        prob_df = prob_df.sort_values('Probability', ascending=False).reset_index(drop=True)
                        fig_bar = go.Figure([go.Bar(x=prob_df['Modulation'], y=prob_df['Probability'], marker_color='#4F8BF9')])
                        fig_bar.update_layout(yaxis=dict(tickformat='.0%'), title='Class Probability Distribution')
                        st.plotly_chart(fig_bar, use_container_width=True)
                        st.dataframe(prob_df, hide_index=True)
                        st.download_button('Download Probabilities as CSV', get_csv_download_link(prob_df, f'{uploaded_file.name}_probs.csv'), file_name=f'{uploaded_file.name}_probs.csv', mime='text/csv')
                    with subtabs[2]:
                        iq_plot = go.Figure()
                        iq_plot.add_trace(go.Scatter(y=iq_data_exp[idx, :, 0], mode='lines', name='I (In-phase)', line=dict(color='#4F8BF9')))
                        iq_plot.add_trace(go.Scatter(y=iq_data_exp[idx, :, 1], mode='lines', name='Q (Quadrature)', line=dict(color='#F97C4F')))
                        iq_plot.update_layout(title='I/Q Waveforms', xaxis_title='Sample Index', yaxis_title='Amplitude')
                        st.plotly_chart(iq_plot, use_container_width=True)
                    with subtabs[3]:
                        fig = plot_constellation(iq_data_exp[idx])
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(get_constellation_image(fig), unsafe_allow_html=True)
                    with subtabs[4]:
                        st.dataframe(stats, hide_index=True)
                    with subtabs[5]:
                        with st.expander("Show Model Summary"):
                            stringlist = []
                            # The original code had model.summary, but model is no longer loaded.
                            # This section will be removed or replaced with a placeholder if model info is needed.
                            pass # Placeholder for model summary
                    with subtabs[6]:
                        snr = estimate_snr(iq_data_exp[idx])
                        st.metric("Estimated SNR (dB)", f"{snr:.2f}" if not np.isnan(snr) else "N/A")
                except Exception as e:
                    st.error(f"Error during prediction or plotting: {e}")
    else:
        st.info('Awaiting file upload.')

    # --- Confusion Matrix (if ground truth provided) ---
    if gt_labels_file is not None and all_preds:
        try:
            gt_labels = np.load(gt_labels_file)
            if len(gt_labels.shape) > 1:
                gt_labels = gt_labels.flatten()
            if len(gt_labels) != len(all_preds):
                st.warning(f"Number of ground truth labels ({len(gt_labels)}) does not match number of predictions ({len(all_preds)}).")
            else:
                cm = confusion_matrix(gt_labels, all_preds, labels=list(range(len(MOD_CLASSES))))
                cm_df = pd.DataFrame(cm, index=MOD_CLASSES, columns=MOD_CLASSES)
                st.subheader("Confusion Matrix")
                st.dataframe(cm_df)
                fig_cm = go.Figure(data=go.Heatmap(z=cm, x=MOD_CLASSES, y=MOD_CLASSES, colorscale='Blues'))
                fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
                st.plotly_chart(fig_cm, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing ground truth labels: {e}")

    # --- Prediction History Tab ---
    if st.session_state['history']:
        st.markdown('---')
        st.subheader('Prediction History (this session)')
        hist_df = pd.DataFrame(st.session_state['history'])
        st.dataframe(hist_df, hide_index=True)
        st.download_button('Download History as CSV', get_csv_download_link(hist_df, 'prediction_history.csv'), file_name='prediction_history.csv', mime='text/csv')

    # --- REST API endpoint (stub) ---
    st.markdown('---')
    st.markdown('**REST API endpoint (stub):** This app can be extended to provide a REST API for programmatic predictions using FastAPI or Flask. Contact the developer for more info.')

    # --- Webhook/email notification (stub) ---
    st.markdown('**Webhook/email notification (stub):** This app can be extended to send notifications on new predictions. Contact the developer for more info.') 