from google import genai
from google.genai import types
import gradio as gr
from google.colab import userdata



client = genai.Client(api_key=userdata.get("GEMINI_API_KEY"))

languages = {
    "Hindi": "Translate the given sentence into Hindi.",
    "Telugu": "Translate the given sentence into Telugu.",
    "French": "Translate the given sentence into French.",
    "Kannada": "Translate the given sentence into Kannada",
    "Tamil": "Translate the given sentence into Tamil",
    "English": "Translate the given sentence into English"
}

def language_translator(question, language):
    system_prompt = languages[language]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
            max_output_tokens=2000
        ),
        contents=question)

    return response.text

demo = gr.Interface(fn = language_translator ,
inputs = [gr.Textbox(lines = 3, placeholder ="Enter a sentence to translate...", label = "Input Sentence" ),
gr.Dropdown( choices=list(languages.keys()), value="Hindi", label="Target Language")],
outputs = gr.Textbox(lines = 5 , label = "Translated Output"),
title = "Language Translator",
description = "Translate a sentence into Hindi, Telugu, or French using Gemini.")


demo.launch(server_name="0.0.0.0", root_path="/gradio" )

question = "how are you "
language = "Telugu"
output = language_translator(question, language)
print(output)
