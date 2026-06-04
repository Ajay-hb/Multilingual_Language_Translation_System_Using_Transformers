# 🌍 Multilingual Language Translation System Using Transformers

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Transformers-NLLB-orange?style=for-the-badge&logo=huggingface">
  <img src="https://img.shields.io/badge/Streamlit-Web_App-red?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/SQLite-Database-blue?style=for-the-badge&logo=sqlite">
  <img src="https://img.shields.io/badge/Deep_Learning-NLP-green?style=for-the-badge">
</p>

## 📌 Overview

The **Multilingual Language Translation System Using Transformers** is an AI-powered Neural Machine Translation (NMT) application designed specifically for Indian languages.

Built using **Meta AI's NLLB (No Language Left Behind) Transformer Model**, the system supports translation between **all 22 Scheduled Languages of India and English**, enabling seamless multilingual communication through text and voice.

The application features:

* 🌐 Multilingual Text Translation
* 🎤 Voice Input (Speech-to-Text)
* 🔍 Automatic Language Detection
* 🔊 Voice Output (Text-to-Speech)
* 🧠 Transformer-Based Neural Machine Translation
* 🗄️ SQLite Translation History
* 📊 Streamlit Interactive Dashboard

---

## 🎯 Project Objectives

* Break language barriers across India.
* Provide accurate multilingual translation using Transformers.
* Support voice-based interaction.
* Enable real-time translation through a user-friendly web application.
* Maintain translation history for future reference.

---

## 🏗️ System Architecture

```text
User Input
    │
    ├── Text Input
    └── Voice Input
            │
            ▼
   Language Detection
            │
            ▼
 Speech-to-Text Module
            │
            ▼
  NLLB Transformer Model
            │
            ▼
     Text Translation
            │
    ┌───────┴────────┐
    ▼                ▼
Text Output    Voice Output
            │
            ▼
      SQLite Database
```

---

## 🚀 Features

### 🌍 Multilingual Translation

Supports translation between:

* English
* Hindi
* Tamil
* Telugu
* Kannada
* Malayalam
* Bengali
* Marathi
* Gujarati
* Punjabi
* Assamese
* Odia
* Sanskrit
* Urdu
* And all remaining Scheduled Languages of India

---

### 🤖 Transformer-Based Translation

The project utilizes **Meta AI's NLLB Transformer Architecture** featuring:

* Encoder
* Decoder
* Self-Attention
* Cross-Attention
* Positional Encoding
* Softmax Prediction Layer

This architecture delivers highly accurate multilingual translations while preserving context and semantics.

---

### 🎤 Voice Input

Convert speech into text using Speech Recognition.

Features:

* Microphone Input
* Automatic Transcription
* Multi-language Speech Support

---

### 🔍 Automatic Language Detection

The system automatically identifies the source language before translation.

Benefits:

* Improved User Experience
* Faster Translation Workflow
* Reduced Manual Configuration

---

### 🔊 Text-to-Speech

Translated text can be converted into natural-sounding speech.

Supports:

* Multiple Indian Languages
* Audio Playback
* Voice Output Download

---

### 🗄️ Translation History

Store and retrieve previous translations using SQLite.

Database stores:

* Source Text
* Source Language
* Target Language
* Translated Text
* Timestamp

---

## 🛠️ Tech Stack

| Category             | Technology                            |
| -------------------- | ------------------------------------- |
| Programming Language | Python                                |
| Deep Learning        | PyTorch                               |
| NLP Framework        | Transformers                          |
| Translation Model    | NLLB                                  |
| Speech Recognition   | SpeechRecognition                     |
| Text-to-Speech       | gTTS                                  |
| Language Detection   | langdetect                            |
| Frontend             | Streamlit                             |
| Database             | SQLite                                |
| Deployment           | Streamlit Cloud / Hugging Face Spaces |

---

## 📂 Project Structure

```text
Multilingual-Translator/
│
├── app.py
├── translator.py
├── speech.py
├── database.py
├── language_detector.py
│
├── models/
│
├── database/
│   └── translations.db
│
├── assets/
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/multilingual-language-translator.git

cd multilingual-language-translator
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Application will launch at:

```text
http://localhost:8501
```

---

## 📈 Future Enhancements

* 📱 Mobile Application
* ☁️ Cloud Deployment
* 🧾 PDF Translation
* 📄 Document Translation
* 🎥 Video Subtitle Translation
* 🧠 Fine-Tuned NLLB Models
* 🌎 Real-Time Translation API

---

## 📊 Skills Demonstrated

* Natural Language Processing (NLP)
* Deep Learning
* Transformer Architecture
* Neural Machine Translation
* Speech Processing
* Language Detection
* Database Management
* Model Deployment
* Streamlit Development
* End-to-End AI Application Development

---

## 💼 Resume Description

Developed an India-focused multilingual language translation system using Meta AI's NLLB Transformer model, supporting translation across all 22 Scheduled Languages of India and English. Integrated automatic language detection, speech-to-text, text-to-speech, and SQLite-based translation history within a Streamlit web application. Implemented Transformer-based neural machine translation leveraging encoder-decoder architecture, self-attention, cross-attention, and sequence-to-sequence learning for accurate real-time multilingual communication.

---

## 👨‍💻 Author

**Ajay Ponnuru**

📧 Email: [your-email@example.com](mailto:your-email@example.com)

🔗 LinkedIn: https://linkedin.com/in/your-profile

🐙 GitHub: https://github.com/Ajay-hb

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub and sharing it with others.

**"Breaking Language Barriers with AI-Powered Translation."**
