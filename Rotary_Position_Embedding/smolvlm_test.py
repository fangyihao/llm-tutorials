from transformers import AutoProcessor, AutoModelForVision2Seq
# from transformers import Idefics3ForConditionalGeneration
from modeling_smolvlm import Idefics3ForConditionalGeneration
from transformers import Idefics3Processor
from transformers import LlamaModel
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = Idefics3Processor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = Idefics3ForConditionalGeneration.from_pretrained("HuggingFaceTB/SmolVLM-Instruct", torch_dtype=torch.bfloat16,
                                                 _attn_implementation="eager").to(DEVICE)
                                              #  _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager").to(DEVICE)

from PIL import Image
from transformers.image_utils import load_image


# Load images
image1 = load_image("https://huggingface.co/spaces/HuggingFaceTB/SmolVLM/resolve/main/example_images/rococo.jpg")
image2 = load_image("https://huggingface.co/spaces/HuggingFaceTB/SmolVLM/resolve/main/example_images/rococo_1.jpg")

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": "Can you describe the two images?"}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")
inputs = inputs.to(DEVICE)


# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
