from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
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
    model_name: str = "facebook/nllb-200-1.3B"
    max_length: int = 256
    num_beams: int = 5
    num_beams: int = 5

    def __post_init__(self):
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
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
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(get_language_code(target_language))
        output_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=self.max_length,
            num_beams=self.num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        return tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        if not text.strip():
            raise TranslationError("Please enter text before translating.")
        if source_language == target_language:
            raise TranslationError("Source and target languages must be different.")
        sentences = self.split_into_sentences(text)
        return " ".join(self.translate_sentence(sentence, source_language, target_language) for sentence in sentences)

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

st.set_page_config(page_title="Multilingual Translator", layout="wide")
st.title("Multilingual Language Translation System")
st.caption("India-focused multilingual translation using Transformers")

history = load_history()
language_names = list(LANGUAGE_OPTIONS.keys())
translate_tab, architecture_tab, history_tab = st.tabs(["Translate", "Architecture", "History"])

with translate_tab:
    source_mode = st.radio("Source language mode", ["Manual selection", "Automatic detection"], horizontal=True)
    st.info("For highest accuracy, keep Manual selection and choose the correct source language. Automatic detection can fail for short or mixed-language text.")
    col1, col2 = st.columns(2)
    with col1:
        source_language = st.selectbox("Source language", language_names)
    with col2:
        target_language = st.selectbox("Target language", language_names, index=1)

    text = st.text_area("Enter text", height=160)

    st.subheader("Voice input")
    speech_language = st.selectbox("Speech language", language_names, key="speech_language")
    st.caption("Use clear audio with low background noise. WAV, AIFF, or FLAC is recommended.")
    audio_file = st.file_uploader("Upload WAV, AIFF, or FLAC audio", type=["wav", "aiff", "aif", "flac"])
    if st.button("Convert voice to text") and audio_file is not None:
        try:
            st.session_state.voice_text = speech_to_text(audio_file, speech_language)
            st.success("Voice converted to text")
        except Exception as exc:
            st.error(f"Speech recognition failed: {exc}")

    if "voice_text" in st.session_state:
        st.text_area("Recognized text", st.session_state.voice_text, height=100)
        if st.button("Use recognized text"):
            text = st.session_state.voice_text

    if st.button("Translate", type="primary"):
        try:
            translator = load_translator()
            for warning in translator.quality_warnings(text, source_mode):
                st.warning(warning)
            actual_source_language = translator.detect_language(text) if source_mode == "Automatic detection" else source_language
            translated_text = translator.translate(text, actual_source_language, target_language)
            history.add(actual_source_language, target_language, text, translated_text)
            st.session_state.latest_translation = translated_text
            st.session_state.latest_target_language = target_language
            st.subheader("Translation")
            st.write(translated_text)
            st.info(f"Source: {actual_source_language} | Target: {target_language}")
        except Exception as exc:
            st.error(f"Translation failed: {exc}")

    if "latest_translation" in st.session_state:
        st.subheader("Voice output")
        if st.button("Generate speech"):
            try:
                audio_bytes = text_to_speech_bytes(
                    st.session_state.latest_translation,
                    st.session_state.latest_target_language,
                )
                st.audio(audio_bytes, format="audio/mp3")
            except Exception as exc:
                st.error(f"Text-to-speech failed: {exc}")

with architecture_tab:
    st.subheader("Project Architecture")
    st.graphviz_chart(ARCHITECTURE_DOT, use_container_width=True)
    st.write("Softmax is used inside attention to create attention weights and at the decoder output to create token probabilities.")

with history_tab:
    st.subheader("Translation History")
    records = history.recent()
    if not records:
        st.write("No translation history yet.")
    for source_language, target_language, source_text, translated_text, created_at in records:
        with st.expander(f"{source_language} to {target_language} - {created_at}"):
            st.write("Input")
            st.write(source_text)
            st.write("Output")
            st.write(translated_text)
    if records and st.button("Clear history"):
        history.clear()
        st.rerun()
