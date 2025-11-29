# app.py
import os
import time
import tempfile
import subprocess
import html
import base64
from typing import List

import streamlit as st
import torch
import whisper

# -------------------------
# Config / Page
# -------------------------
APP_TITLE = "IELTS Speaking Transcriber"
APP_AUTHOR = "HoangIELTS"
PRIMARY_BLUE = "#0b63b7"  # deep ocean blue
ACCENT = "#0b84d6"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="auto",
)

# -------------------------
# Visual theme (CSS) with safe token replacement
# -------------------------
_CUSTOM_CSS = """
<style>
/* page background */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #f6f9fc 0%, #ffffff 100%);
  color: #0b2b45;
}

/* hide default Streamlit header menu */
header .css-1v3fvcr {
  visibility: hidden;
}

/* Title style */
h1 {
  color: <<PRIMARY>>;
  margin-bottom: 6px;
}

/* Buttons */
.stButton>button {
  background-color: <<PRIMARY>>;
  color: white;
  border-radius: 8px;
  padding: 8px 14px;
  border: none;
  font-weight: 600;
}
.stButton>button:hover {
  background-color: <<ACCENT>>;
  transform: translateY(-1px);
  transition: all 0.08s ease-in-out;
}

/* Download button style */
div.stDownloadButton > button {
  background-color: <<ACCENT>>;
  color: white;
  border-radius: 8px;
}

/* Sidebar style */
[data-testid="stSidebar"] {
  background-color: #f3f8ff;
  border-right: 1px solid rgba(11,99,183,0.06);
}

/* small helper */
.small-muted {
  color: #6b7280;
  font-size: 13px;
}

/* compact spacing for header area */
.block-container .css-1d391kg {
  padding-top: 16px;
}

/* make code / transcript area monospaced */
textarea[aria-label="Transcript"] {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace;
  font-size: 14px;
}
</style>
"""

# replace tokens safely
_CUSTOM_CSS = _CUSTOM_CSS.replace("<<PRIMARY>>", PRIMARY_BLUE).replace("<<ACCENT>>", ACCENT)
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------
# FFmpeg local support
# -------------------------
proj_dir = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
local_ffmpeg_bin = os.path.join(proj_dir, "ffmpeg", "bin")
if os.path.isdir(local_ffmpeg_bin):
    os.environ["PATH"] = local_ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

ffmpeg_exe = os.path.join(local_ffmpeg_bin, "ffmpeg.exe")
ffprobe_exe = os.path.join(local_ffmpeg_bin, "ffprobe.exe")
if os.path.exists(ffmpeg_exe):
    os.environ["FFMPEG_BINARY"] = ffmpeg_exe
if os.path.exists(ffprobe_exe):
    os.environ["FFPROBE_BINARY"] = ffprobe_exe


def check_ffmpeg() -> bool:
    try:
        proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        return proc.returncode == 0
    except FileNotFoundError:
        return False
    except Exception:
        return False


# -------------------------
# Model loader (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model_cached(model_name: str = "base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name)
    return model, device


def transcribe(audio_path: str, model, fp16: bool, with_timestamps: bool):
    result = model.transcribe(audio_path, fp16=fp16)
    text = result.get("text", "").strip()
    segments = result.get("segments", []) if with_timestamps else []
    return text, segments


def format_segments_to_text(segments: List[dict]) -> str:
    lines = []
    for seg in segments:
        s = seg.get("start", 0.0)
        e = seg.get("end", 0.0)
        t = seg.get("text", "").strip()
        lines.append("[{:.2f}s -> {:.2f}s] {}".format(s, e, t))
    return "\n".join(lines)


def segments_to_srt(segments: List[dict]) -> str:
    def to_timestamp(sec):
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec - int(sec)) * 1000)
        return "{:02d}:{:02d}:{:02d},{:03d}".format(h, m, s, ms)

    parts = []
    for i, seg in enumerate(segments, start=1):
        start = to_timestamp(seg.get("start", 0.0))
        end = to_timestamp(seg.get("end", 0.0))
        text = seg.get("text", "").strip()
        parts.append("{}\n{} --> {}\n{}\n".format(i, start, end, text))
    return "\n".join(parts)


# -------------------------
# Sidebar (menu + author)
# -------------------------
st.sidebar.title(APP_TITLE)
st.sidebar.markdown(f"***{APP_AUTHOR}***")
st.sidebar.markdown("---")
menu_choice = st.sidebar.selectbox("Menu", ["Transcribe", "Instructions", "About"])

if menu_choice == "Instructions":
    st.sidebar.markdown(
        """
- Upload audio (mp3/wav/m4a/ogg/flac).
- Choose *Only text* or *With timestamps*.
- Click **Process and Convert**.
- Edit transcript in the box, then download or copy to clipboard.
"""
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Tip:** For long files, use a GPU-enabled machine and a larger model size (small/medium).")
elif menu_choice == "About":
    st.sidebar.markdown(
        """
**{}**  
Minimal transcription UI powered by Whisper.  
Created by **{}**.
""".format(APP_TITLE, APP_AUTHOR)
    )
    st.sidebar.markdown("---")


# -------------------------
# Main UI
# -------------------------
st.header(APP_TITLE)
st.caption("Simple, minimal, modern transcription interface built by Hoang.")

col_left, col_right = st.columns([1, 2])

with col_left:
    uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a", "flac", "ogg", "webm"])
    mode = st.radio("Mode", options=["Only text", "With timestamps"], index=0)
    # Restrict model choices to tiny or base only
    model_name = st.selectbox("Model size", options=["tiny", "base"], index=1)
    start_btn = st.button("Process and Convert")
    st.markdown("<div class='small-muted'>Model size affects accuracy and resource usage.</div>", unsafe_allow_html=True)

with col_right:
    st.info("If you have GPU + CUDA, transcription is faster. The app will use fp16 on GPU automatically.")
    if not check_ffmpeg():
        st.warning("ffmpeg not found. Place ffmpeg/bin next to app.py or install ffmpeg system-wide.")

# Load model UI (cached)
with st.spinner("Loading model..."):
    try:
        model, device = load_model_cached(model_name)
        st.success("Loaded model: {} on device: {}".format(model_name, device))
    except Exception as e:
        st.error("Error loading model {}: {}".format(model_name, e))
        st.stop()

if start_btn:
    if uploaded_file is None:
        st.warning("Please upload an audio file before starting.")
    else:
        if not check_ffmpeg():
            st.error("ffmpeg not found. Install or place ffmpeg/bin next to app.py and restart.")
            st.stop()

        # Save uploaded file to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.info("Processing: {}  (model={})".format(uploaded_file.name, model_name))
        fp16 = True if torch.cuda.is_available() else False

        # debug: show tmp path
        st.write("Temporary audio path: `{}`".format(tmp_path))

        t0 = time.time()
        try:
            text, segments = transcribe(tmp_path, model, fp16=fp16, with_timestamps=(mode == "With timestamps"))
        except Exception as e:
            st.error("Error during transcription: {}".format(e))
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            st.stop()

        elapsed = time.time() - t0
        st.success("Transcription finished in {:.1f}s".format(elapsed))

        # Display segments and transcript
        if mode == "With timestamps" and segments:
            st.subheader("Segments (table)")
            import pandas as pd

            df = pd.DataFrame([{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in segments])
            st.dataframe(df, use_container_width=True)

            st.subheader("Segments (plain with timestamps)")
            seg_plain = format_segments_to_text(segments)
            st.code(seg_plain, language=None)

            st.subheader("Merged Transcript (editable)")
            default_text = text
            textarea_value = st.text_area("Transcript", value=default_text, height=300, key="transcript_area")
        else:
            st.subheader("Transcript (editable)")
            default_text = text
            textarea_value = st.text_area("Transcript", value=default_text, height=400, key="transcript_area")

        final_text = textarea_value if textarea_value is not None else default_text

        # Downloads and SRT
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.download_button("Download .txt", data=final_text, file_name="transcript.txt", mime="text/plain")
        with col2:
            if mode == "With timestamps" and segments:
                srt_text = segments_to_srt(segments)
                st.download_button("Download .srt", data=srt_text, file_name="transcript.srt", mime="text/plain")
            else:
                st.button("No SRT for Only text", disabled=True)

        # Copy to clipboard — use base64 to avoid escaping issues
        b64 = base64.b64encode(final_text.encode("utf-8")).decode("utf-8")
        copy_html = """
        <div style="display:flex; gap:10px; align-items:center;">
            <button id="copyBtn" style="background:%s;color:white;padding:8px 12px;border-radius:8px;border:none;">
                Copy to clipboard
            </button>
            <span id="copyStatus" style="color:#0b2b45;font-size:14px;margin-left:6px;"></span>
        </div>
        <script>
        const b64 = "%s";
        const btn = document.getElementById("copyBtn");
        const status = document.getElementById("copyStatus");
        btn.addEventListener("click", async () => {
            try {
                const decoded = atob(b64);
                await navigator.clipboard.writeText(decoded);
                status.textContent = "Copied!";
            } catch (err) {
                status.textContent = "Copy failed. Select and copy manually.";
            }
        });
        </script>
        """ % (PRIMARY_BLUE, b64)

        st.components.v1.html(copy_html, height=70)

        # Clean up tmp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass

st.markdown("---")
st.caption("© {}  •  Built with Whisper + Streamlit".format(APP_AUTHOR))
