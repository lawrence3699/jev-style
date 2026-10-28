"""Reproduce the synthetic receipt fixture; requires Pillow. No real customer data."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

canvas = Image.new("RGB", (600, 800), "white")
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default(size=24)
heading = ImageFont.load_default(size=30)
draw.text((40, 40), "SYNTHETIC SAMPLE", fill="black", font=heading)
for i, line in enumerate([
    "Example Cafe", "MEAL RECEIPT", "Receipt: DEMO-001", "Currency: EUR", "",
    "Lunch                 36.00", "Dessert               12.00", "Drinks                12.00", "",
    "TOTAL                 60.00", "", "For software testing only.", "Not a valid tax invoice.",
]):
    draw.text((40, 100 + i * 44), line, fill="black", font=font)
canvas.save(Path(__file__).with_name("synthetic_receipt.png"))
