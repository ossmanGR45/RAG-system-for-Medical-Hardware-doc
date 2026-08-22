import fitz  # PyMuPDF

pdf_path = "data/manuals/SM SLX-F2(25k) 0516.pdf"
doc = fitz.open(pdf_path)

extracted_text = ""
for page_num in range(min(5, len(doc))):
    extracted_text += f"--- PAGE {page_num + 1} ---\n"
    extracted_text += doc[page_num].get_text("text") + "\n\n"

with open("extracted_sample.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)

print("Saved sample to extracted_sample.txt")