from google import genai
from google.genai import types
from dotenv import load_dotenv
import requests
import json
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
