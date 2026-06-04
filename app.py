from __future__ import annotations

import sqlite3
from time import perf_counter
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path

import streamlit as st

LANGUAGE_OPTIONS = {
    "English": "eng_Latn",
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Kashmiri": "kas_Arab",
    "Konkani": "gom_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Manipuri": "mni_Beng",
    "Marathi": "mar_Deva",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi": "snd_Arab",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab",
}

DETECTION_CODES = {
    "en": "English",
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}
TTS_CODES = {
    "English": "en",
    "Assamese": "as",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Nepali": "ne",
    "Punjabi": "pa",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
}

ARCHITECTURE_DOT = 'digraph translation_architecture {\n    graph [rankdir=TB, bgcolor="transparent", pad="0.3", nodesep="0.45", ranksep="0.55"];\n    node [shape=box, style="rounded,filled", color="#506070", fillcolor="#F8FAFC", fontname="Arial", fontsize=10];\n    edge [color="#64748B", arrowsize=0.8];\n\n    user [label="User"];\n    app [label="Streamlit Web Application", fillcolor="#E0F2FE"];\n    text_input [label="Text Input"];\n    voice_input [label="Voice Input"];\n    stt [label="Speech-to-Text"];\n    source_text [label="Source Text"];\n    language [label="Manual Language Selection / Auto Detection"];\n    tokenizer [label="Tokenizer"];\n    encoder [label="Transformer Encoder"];\n    attention_scores [label="Attention Scores"];\n    attention_softmax [label="Softmax"];\n    attention_weights [label="Attention Weights"];\n    decoder [label="Transformer Decoder"];\n    cross_attention [label="Cross-Attention"];\n    output_scores [label="Output Token Scores"];\n    output_softmax [label="Softmax"];\n    probabilities [label="Token Probabilities"];\n    translated [label="Translated Text", fillcolor="#DCFCE7"];\n    tts [label="Text-to-Speech"];\n    audio [label="Voice Output"];\n    db [label="SQLite Translation History"];\n    history [label="History View"];\n\n    user -> app;\n    app -> text_input;\n    app -> voice_input;\n    voice_input -> stt -> source_text;\n    text_input -> source_text;\n    source_text -> language -> tokenizer -> encoder;\n    encoder -> attention_scores -> attention_softmax -> attention_weights -> decoder;\n    decoder -> cross_attention -> output_scores -> output_softmax -> probabilities -> translated;\n    translated -> tts -> audio;\n    translated -> db -> history;\n    translated -> app;\n    audio -> app;\n    history -> app;\n}'

def get_language_code(language_name: str) -> str:
    return LANGUAGE_OPTIONS[language_name]

def get_tts_code(language_name: str) -> str:
    return TTS_CODES[language_name]

def detect_supported_language(text: str) -> str:
    from langdetect import detect
    detected_code = detect(text)
    if detected_code not in DETECTION_CODES:
        raise ValueError("Automatic detection is best-effort. Use manual selection if detection fails.")
    return DETECTION_CODES[detected_code]

class TranslationError(RuntimeError):
    pass

@dataclass(slots=True)
class TransformerTranslator:
    model_name: str = "facebook/nllb-200-distilled-600M"
    max_length: int = 64
    num_beams: int = 1
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)

    def _load_model(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.eval()
        except Exception as exc:
            raise TranslationError("Model could not be loaded. Check internet access or local model cache.") from exc
        return self._tokenizer, self._model

    def detect_language(self, text: str) -> str:
        try:
            return detect_supported_language(text)
        except Exception as exc:
            raise TranslationError("Automatic language detection failed. Select the source language manually.") from exc

    def split_into_sentences(self, text: str):
        import re
        cleaned_text = " ".join(text.split())
        return [part.strip() for part in re.split(r"(?<=[.!??])\s+", cleaned_text) if part.strip()]

    def quality_warnings(self, text: str, source_mode: str):
        warnings = []
        if len(text.split()) < 3:
            warnings.append("Input is very short, so translation may be less reliable.")
        if source_mode == "Automatic detection":
            warnings.append("Manual source language selection gives better accuracy.")
        if len(text) > 700:
            warnings.append("Long text will be split into sentences for better translation quality.")
        return warnings

    def translate_sentence(self, text: str, source_language: str, target_language: str) -> str:
        tokenizer, model = self._load_model()
        tokenizer.src_lang = get_language_code(source_language)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(get_language_code(target_language))
        import torch
        torch.set_num_threads(2)

        with torch.inference_mode():
            output_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=self.max_length,
                num_beams=self.num_beams,
            )
        return tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        if not text.strip():
            raise TranslationError("Please enter text before translating.")
        if len(text.split()) > 25:
            raise TranslationError("For under-5-second demo translation, enter 25 words or fewer.")
        if source_language == target_language:
            raise TranslationError("Source and target languages must be different.")
        # Fast demo mode: translate the input in one model call.
        return self.translate_sentence(text, source_language, target_language)

class TranslationHistory:
    def __init__(self, database_path: str | Path = "translation_history.db"):
        self.database_path = Path(database_path)
        self._create_table()

    def _create_table(self):
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS translation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_language TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

    def add(self, source_language: str, target_language: str, source_text: str, translated_text: str):
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO translation_history
                (source_language, target_language, source_text, translated_text, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source_language,
                    target_language,
                    source_text,
                    translated_text,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )

    def recent(self, limit: int = 20):
        with sqlite3.connect(self.database_path) as connection:
            return connection.execute(
                """
                SELECT source_language, target_language, source_text, translated_text, created_at
                FROM translation_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def clear(self):
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("DELETE FROM translation_history")

def speech_to_text(audio_file, language_name: str) -> str:
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    with sr.AudioFile(audio_file) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio, language=get_tts_code(language_name))

def text_to_speech_bytes(text: str, language_name: str) -> bytes:
    from gtts import gTTS
    buffer = BytesIO()
    gTTS(text=text, lang=get_tts_code(language_name)).write_to_fp(buffer)
    return buffer.getvalue()

@st.cache_resource(show_spinner="Loading Transformer model...")
def load_translator():
    return TransformerTranslator()

@st.cache_resource
def load_history():
    return TranslationHistory()

def apply_theme():
    st.markdown(
        """
        <style>
        :root {
            --surface: #ffffff;
            --surface-soft: #f8fafc;
            --line: #d8e0ea;
            --ink: #0f172a;
            --muted: #5f6f84;
            --brand: #0f766e;
            --brand-dark: #115e59;
            --accent: #2563eb;
            --success-bg: #ecfdf5;
            --success-line: #a7f3d0;
            --warn-bg: #fff7ed;
            --warn-line: #fed7aa;
        }
        .main .block-container {
            max-width: 1220px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        .app-hero {
            border: 1px solid var(--line);
            background:
                radial-gradient(circle at 14% 18%, rgba(20, 184, 166, .22), transparent 28%),
                radial-gradient(circle at 92% 12%, rgba(37, 99, 235, .20), transparent 30%),
                linear-gradient(135deg, #f8fafc 0%, #ecfeff 48%, #eef2ff 100%);
            background-size: 160% 160%;
            padding: 28px 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
            animation: heroFlow 9s ease-in-out infinite alternate;
            box-shadow: 0 18px 45px rgba(15, 23, 42, .08);
        }
        .app-hero::after {
            content: "";
            position: absolute;
            inset: auto -8% -35% 45%;
            height: 160px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,.65), transparent);
            transform: rotate(-8deg);
            animation: sheen 5s ease-in-out infinite;
        }
        .app-title {
            color: var(--ink);
            font-size: 34px;
            font-weight: 800;
            margin: 0;
            position: relative;
            z-index: 1;
        }
        .app-subtitle {
            color: var(--muted);
            font-size: 16px;
            margin-top: 8px;
            max-width: 860px;
            line-height: 1.55;
            position: relative;
            z-index: 1;
        }
        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 16px 0 8px;
        }
        .metric-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
            animation: riseIn .5s ease both;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            border-color: rgba(15, 118, 110, .35);
            box-shadow: 0 14px 34px rgba(15, 23, 42, .10);
        }
        .metric-label {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .06em;
            font-weight: 750;
        }
        .metric-value {
            color: var(--ink);
            font-size: 20px;
            font-weight: 800;
            margin-top: 4px;
        }
        .section-panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 16px;
            animation: riseIn .55s ease both;
            box-shadow: 0 10px 30px rgba(15, 23, 42, .05);
        }
        .panel-title {
            color: var(--ink);
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 6px;
        }
        .panel-help {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 14px;
        }
        .output-box {
            background: var(--success-bg);
            border: 1px solid var(--success-line);
            border-radius: 8px;
            padding: 18px;
            margin-top: 16px;
            position: relative;
            overflow: hidden;
            animation: outputPop .38s ease both;
        }
        .output-box::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 5px;
            background: linear-gradient(#10b981, #0f766e);
            animation: pulseBar 1.9s ease-in-out infinite;
        }
        .output-label {
            color: #166534;
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .06em;
            margin-bottom: 8px;
        }
        .output-text {
            color: #052e16;
            font-size: 21px;
            line-height: 1.65;
            font-weight: 650;
            word-break: break-word;
        }
        .status-pill {
            display: inline-block;
            background: #e0f2fe;
            color: #075985;
            border: 1px solid #bae6fd;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 13px;
            font-weight: 750;
            margin-top: 10px;
            animation: softPulse 2.4s ease-in-out infinite;
        }
        .stButton > button {
            border-radius: 7px;
            font-weight: 750;
            transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(15, 118, 110, .18);
        }
        .stTextArea textarea, .stSelectbox div[data-baseweb="select"], .stFileUploader section {
            transition: border-color .18s ease, box-shadow .18s ease;
        }
        .stTextArea textarea:focus {
            border-color: rgba(15, 118, 110, .55) !important;
            box-shadow: 0 0 0 3px rgba(15, 118, 110, .12) !important;
        }
        div[data-testid="stTabs"] button {
            font-weight: 750;
            transition: color .18s ease, background .18s ease;
        }
        div[data-testid="stTabs"] button:hover {
            color: var(--brand-dark);
        }
        div[data-testid="stTabs"] [aria-selected="true"] {
            color: var(--brand-dark);
        }
        @keyframes heroFlow {
            from { background-position: 0% 50%; }
            to { background-position: 100% 50%; }
        }
        @keyframes sheen {
            0%, 35% { transform: translateX(-70%) rotate(-8deg); opacity: 0; }
            50% { opacity: .75; }
            100% { transform: translateX(70%) rotate(-8deg); opacity: 0; }
        }
        @keyframes riseIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes outputPop {
            from { opacity: 0; transform: scale(.985) translateY(8px); }
            to { opacity: 1; transform: scale(1) translateY(0); }
        }
        @keyframes pulseBar {
            0%, 100% { opacity: .55; }
            50% { opacity: 1; }
        }
        @keyframes softPulse {
            0%, 100% { box-shadow: 0 0 0 rgba(14, 165, 233, 0); }
            50% { box-shadow: 0 0 0 5px rgba(14, 165, 233, .10); }
        }
        @media (max-width: 760px) {
            .metric-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .app-title {
                font-size: 27px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Multilingual Translator", layout="wide")
apply_theme()
st.markdown(
    """
    <div class="app-hero">
        <div class="app-title">Multilingual Language Translation System</div>
        <div class="app-subtitle">
            A Transformer-based translation workspace for Indian languages with text translation,
            speech input, voice output, architecture visualization, and SQLite translation history.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
with metric_cols[0]:
    render_metric("Languages", "22 + English")
with metric_cols[1]:
    render_metric("Model", "NLLB Distilled")
with metric_cols[2]:
    render_metric("Mode", "Fast Demo")
with metric_cols[3]:
    render_metric("Storage", "SQLite")

history = load_history()
language_names = list(LANGUAGE_OPTIONS.keys())
translate_tab, architecture_tab, history_tab = st.tabs(["Translate", "Architecture", "History"])

with translate_tab:
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Translation Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-help">Fast demo mode is optimized for short inputs. Use 25 words or fewer after the model has loaded.</div>',
        unsafe_allow_html=True,
    )

    mode_col, source_col, target_col = st.columns([1.2, 1, 1])
    with mode_col:
        source_mode = st.radio(
            "Source mode",
            ["Manual selection", "Automatic detection"],
            horizontal=True,
        )
    with source_col:
        source_language = st.selectbox("Source language", language_names)
    with target_col:
        target_language = st.selectbox("Target language", language_names, index=1)

    input_col, tools_col = st.columns([1.45, 1])
    with input_col:
        text = st.text_area(
            "Text to translate",
            height=190,
            placeholder="Enter a short sentence for faster translation...",
        )
        word_count = len(text.split())
        st.caption(f"Word count: {word_count}/25")
        translate_clicked = st.button("Translate Text", type="primary", use_container_width=True)

    with tools_col:
        st.markdown('<div class="panel-title">Voice Input</div>', unsafe_allow_html=True)
        speech_language = st.selectbox("Speech language", language_names, key="speech_language")
        st.caption("Use clear WAV, AIFF, or FLAC audio with low background noise.")
        audio_file = st.file_uploader(
            "Upload audio",
            type=["wav", "aiff", "aif", "flac"],
        )
        if st.button("Convert Voice To Text", use_container_width=True) and audio_file is not None:
            try:
                st.session_state.voice_text = speech_to_text(audio_file, speech_language)
                st.success("Voice converted to text.")
            except Exception as exc:
                st.error(f"Speech recognition failed: {exc}")

        if "voice_text" in st.session_state:
            st.text_area("Recognized text", st.session_state.voice_text, height=92)
            if st.button("Use Recognized Text", use_container_width=True):
                text = st.session_state.voice_text

    if translate_clicked:
        try:
            translator = load_translator()
            for warning in translator.quality_warnings(text, source_mode):
                st.warning(warning)
            actual_source_language = translator.detect_language(text) if source_mode == "Automatic detection" else source_language
            started_at = perf_counter()
            translated_text = translator.translate(text, actual_source_language, target_language)
            elapsed = perf_counter() - started_at
            history.add(actual_source_language, target_language, text, translated_text)
            st.session_state.latest_translation = translated_text
            st.session_state.latest_target_language = target_language
            st.session_state.latest_source_language = actual_source_language
            st.session_state.latest_elapsed = elapsed
        except Exception as exc:
            st.error(f"Translation failed: {exc}")

    if "latest_translation" in st.session_state:
        safe_translation = escape(st.session_state.latest_translation)
        safe_source = escape(st.session_state.latest_source_language)
        safe_target = escape(st.session_state.latest_target_language)
        st.markdown(
            f"""
            <div class="output-box">
                <div class="output-label">Translated Text</div>
                <div class="output-text">{safe_translation}</div>
                <div class="status-pill">
                    {safe_source} to {safe_target}
                    | {st.session_state.latest_elapsed:.2f}s after model load
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        voice_col, copy_col = st.columns([1, 1])
        with voice_col:
            if st.button("Generate Voice Output", use_container_width=True):
                try:
                    audio_bytes = text_to_speech_bytes(
                        st.session_state.latest_translation,
                        st.session_state.latest_target_language,
                    )
                    st.audio(audio_bytes, format="audio/mp3")
                except Exception as exc:
                    st.error(f"Text-to-speech failed: {exc}")
        with copy_col:
            st.download_button(
                "Download Translation",
                data=st.session_state.latest_translation,
                file_name="translation.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

with architecture_tab:
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Project Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-help">The model uses encoder-decoder Transformer layers with attention, cross-attention, and Softmax.</div>',
        unsafe_allow_html=True,
    )
    st.graphviz_chart(ARCHITECTURE_DOT, use_container_width=True)
    st.info("Softmax is used inside attention to create attention weights and at the decoder output to create token probabilities.")
    st.markdown("</div>", unsafe_allow_html=True)

with history_tab:
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Translation History</div>', unsafe_allow_html=True)
    records = history.recent()
    if not records:
        st.info("No translation history yet.")
    for source_language, target_language, source_text, translated_text, created_at in records:
        with st.expander(f"{source_language} to {target_language} - {created_at}"):
            left, right = st.columns(2)
            with left:
                st.caption("Input")
                st.write(source_text)
            with right:
                st.caption("Output")
                st.write(translated_text)
    if records and st.button("Clear history"):
        history.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
