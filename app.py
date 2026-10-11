
import os
import tempfile

import gradio as gr
from google import genai
from google.genai import types
from gtts import gTTS

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

LANGUAGES = [
    "English", "Hindi", "Kannada",
    "Telugu", "Tamil", "French"
]

LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Telugu": "te",
    "Tamil": "ta",
    "French": "fr",
}


def translate_text(text, target_language):
    if not text or not text.strip():
        raise gr.Error("Please enter some text.")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=(
                f"Translate the input into {target_language}. "
                "Return only the translation, preserving the "
                "original meaning and tone."
            ),
            temperature=0.3,
            max_output_tokens=2000,
        ),
        contents=text.strip(),
    )

    if not response.text:
        raise gr.Error("No translation was returned.")

    return response.text.strip()


def generate_speech(text, target_language):
    if not text or not text.strip():
        raise gr.Error("There is no translated text to speak.")

    path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        ) as audio_file:
            path = audio_file.name

        gTTS(
            text=text,
            lang=LANGUAGE_CODES[target_language],
        ).save(path)

        return path

    except Exception as exc:
        if path and os.path.exists(path):
            os.remove(path)

        raise gr.Error(
            f"Speech generation failed: {exc}"
        )


def text_to_speech(text, target_language):
    translated = translate_text(text, target_language)
    audio = generate_speech(translated, target_language)
    return translated, audio


def speech_to_speech(audio_path, target_language):
    if not audio_path:
        raise gr.Error("Please record or upload an audio file.")

    uploaded_file = None

    try:
        uploaded_file = client.files.upload(file=audio_path)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Listen to the supplied audio, understand "
                    "the spoken words, and translate them into "
                    f"{target_language}. Return only the "
                    "translated text."
                ),
                temperature=0.3,
                max_output_tokens=2000,
            ),
            contents=[
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=uploaded_file.mime_type,
                )
            ],
        )

        if not response.text:
            raise gr.Error("Could not translate the audio.")

        translated = response.text.strip()

    except gr.Error:
        raise

    except Exception as exc:
        raise gr.Error(f"Audio translation failed: {exc}")

    finally:
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

    audio = generate_speech(translated, target_language)
    return translated, audio


with gr.Blocks(title="AI Language Translator") as demo:
    gr.Markdown(
        "# 🌐 AI Language Translator\n"
        "Translate text and speech into six languages."
    )

    with gr.Tab("Text → Text"):
        input_text = gr.Textbox(
            label="Enter text",
            lines=3,
            placeholder="Type a sentence to translate...",
        )
        target = gr.Dropdown(
            choices=LANGUAGES,
            value="Kannada",
            label="Target language",
        )
        translate_button = gr.Button(
            "Translate", variant="primary"
        )
        translated_text = gr.Textbox(
            label="Translated text", lines=4
        )

        translate_button.click(
            translate_text,
            inputs=[input_text, target],
            outputs=translated_text,
        )

    with gr.Tab("Text → Speech"):
        tts_input = gr.Textbox(
            label="Enter text",
            lines=3,
            placeholder="Type text to translate and hear...",
        )
        tts_target = gr.Dropdown(
            choices=LANGUAGES,
            value="Kannada",
            label="Target language",
        )
        tts_button = gr.Button(
            "Translate and Speak", variant="primary"
        )
        tts_text = gr.Textbox(label="Translated text")
        tts_audio = gr.Audio(
            label="Translated speech",
            type="filepath",
        )

        tts_button.click(
            text_to_speech,
            inputs=[tts_input, tts_target],
            outputs=[tts_text, tts_audio],
        )

    with gr.Tab("Speech → Speech"):
        speech_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Record or upload speech",
        )
        speech_target = gr.Dropdown(
            choices=LANGUAGES,
            value="Kannada",
            label="Target language",
        )
        speech_button = gr.Button(
            "Translate Speech", variant="primary"
        )
        speech_text = gr.Textbox(label="Translated text")
        speech_audio = gr.Audio(
            label="Translated speech",
            type="filepath",
        )

        speech_button.click(
            speech_to_speech,
            inputs=[speech_input, speech_target],
            outputs=[speech_text, speech_audio],
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
