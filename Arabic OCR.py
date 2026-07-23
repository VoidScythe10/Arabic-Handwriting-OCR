import os
import json
import logging
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from natsort import natsorted
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from qwen_vl_utils import process_vision_info
from huggingface_hub import login

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_RIGHT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import arabic_reshaper
from bidi.algorithm import get_display

# =====================================================================
#   CONFIG (EDIT PATHS)
# =====================================================================
IMAGE_DIR      = r"C:\Users\Abdullah\Pictures\OCR"         # folder containing your images
OUTPUT_PDF     = r"C:\Users\Abdullah\Desktop\output.pdf"           # where to save the final PDF
PROGRESS_FILE  = r"C:\Users\Abdullah\Desktop\ocr_progress.json"   # auto-resume file (created automatically)
ARABIC_FONT    = r"C:\Users\Abdullah\fonts\Amiri-Regular.ttf"     # path to your Arabic .ttf font
HF_TOKEN       = os.environ.get("HF_TOKEN", "")                   # set HF_TOKEN env variable, or paste token here as fallback

MODEL_PATH     = "sherif1313/Arabic-Qwen3.5-OCR-v4"
OCR_PROMPT     = "اقرأ النص في هذه الصورة كاملاً من البداية إلى النهاية."
IMAGE_EXTS     = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
IMAGES_PER_PAGE = 1   # how many images per PDF page before a page break
# =====================================================================


# ==================== Setup ====================
logging.getLogger("transformers").setLevel(logging.ERROR)

if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("[WARN] No HuggingFace token found. Set the HF_TOKEN environment variable.")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
print(f"[INFO] Using device: {device}")


# ==================== Load Model ====================
print("[INFO] Loading model — this may take a minute...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True
)
model.eval()
print("[INFO] Model ready!")


# ==================== OCR ====================
def extract_text(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    w, h  = image.size
    image = image.resize(
        (((w + 63) // 64) * 64, ((h + 63) // 64) * 64),
        Image.LANCZOS
    )

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": OCR_PROMPT}
    ]}]

    text_input      = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    input_len = inputs.input_ids.shape[1]
    return processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()


# ==================== Arabic PDF helpers ====================
def to_rtl(text: str) -> str:
    lines = text.splitlines()
    processed = [
        get_display(arabic_reshaper.reshape(line)) if line.strip() else ""
        for line in lines
    ]
    return "<br/>".join(processed)


def build_pdf(results: list, output_path: str):
    print("[INFO] Building PDF...")

    pdfmetrics.registerFont(TTFont("Arabic", ARABIC_FONT))

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    title_style = ParagraphStyle("Title",
        fontName="Arabic", fontSize=20,
        alignment=TA_CENTER, spaceAfter=16, leading=32)

    label_style = ParagraphStyle("Label",
        fontName="Arabic", fontSize=9,
        alignment=TA_RIGHT, textColor="grey",
        spaceBefore=14, spaceAfter=4)

    body_style = ParagraphStyle("Body",
        fontName="Arabic", fontSize=13,
        alignment=TA_RIGHT, leading=24,
        wordWrap="RTL", spaceAfter=8)

    error_style = ParagraphStyle("Error",
        fontName="Arabic", fontSize=11,
        alignment=TA_RIGHT, textColor="red")

    story = [
        Paragraph(to_rtl("نتائج استخراج النصوص"), title_style),
        HRFlowable(width="100%", thickness=1.2, color="black"),
        Spacer(1, 0.5 * cm),
    ]

    for i, (filename, text) in enumerate(results):
        # Image label
        story.append(Paragraph(to_rtl(f"الصورة {i + 1}") + f": {filename}", label_style))
        story.append(HRFlowable(width="100%", thickness=0.4, color="lightgrey"))
        story.append(Spacer(1, 0.15 * cm))

        # Extracted text (or error note)
        if text:
            story.append(Paragraph(to_rtl(text), body_style))
        else:
            story.append(Paragraph(to_rtl("[ تعذّر استخراج النص ]"), error_style))

        story.append(Spacer(1, 0.4 * cm))

        # Page break every N images (but not after the very last one)
        if (i + 1) % IMAGES_PER_PAGE == 0 and i < len(results) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"[✅] PDF saved → {output_path}")


# ==================== Progress (auto-resume) ====================
def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ==================== Main ====================
if __name__ == "__main__":

    # --- Find & sort images (natural order: scan_1, scan_2, ... scan_10, not scan_1, scan_10, scan_2) ---
    all_images = natsorted(
        [p for p in Path(IMAGE_DIR).iterdir() if p.suffix.lower() in IMAGE_EXTS]
    )

    if not all_images:
        print(f"[❌] No images found in: {IMAGE_DIR}")
        exit()

    print(f"[INFO] Found {len(all_images)} images (natural sort order).")
    print(f"[INFO] First: {all_images[0].name}  |  Last: {all_images[-1].name}")

    # --- Load previous progress (skip already-done images if resuming) ---
    progress = load_progress()
    already_done = sum(1 for v in progress.values() if v != "__error__")
    if already_done:
        print(f"[INFO] Resuming — {already_done} images already processed, skipping them.")

    results = []
    failed  = []

    # --- Process each image ---
    for img_path in tqdm(all_images, desc="OCR Progress", unit="img"):
        fname = img_path.name

        # Use cached result if available
        if fname in progress:
            cached = progress[fname]
            results.append((fname, "" if cached == "__error__" else cached))
            continue

        # Run OCR
        try:
            text = extract_text(str(img_path))
            progress[fname] = text
            results.append((fname, text))
        except Exception as e:
            tqdm.write(f"[WARN] Failed on {fname}: {e}")
            progress[fname] = "__error__"
            results.append((fname, ""))
            failed.append(fname)

        # Save after every image so a crash loses nothing
        save_progress(progress)

    # --- Summary ---
    print(f"\n[INFO] Done — {len(results)} images processed, {len(failed)} failed.")
    if failed:
        print(f"[WARN] Failed images: {failed}")

    # --- Build the PDF ---
    build_pdf(results, OUTPUT_PDF)