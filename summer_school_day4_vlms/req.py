from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image

from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import ollama