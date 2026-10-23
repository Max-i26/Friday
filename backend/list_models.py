import os
import google.generativeai as genai
from dotenv import load_dotenv

def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        try:
            print("Listing supported models for your API key:")
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(f" - {m.name}")
        except Exception as e:
            print("Error listing models:", e)
    else:
        print("No GEMINI_API_KEY in environment.")

if __name__ == "__main__":
    main()
