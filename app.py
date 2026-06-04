
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
