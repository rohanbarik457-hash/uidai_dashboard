"""
Professional PDF Generator for UIDAI Hackathon Submission
With embedded images and colorful tricolor theme
"""

from markdown_pdf import MarkdownPdf, Section
import re
import os

# Read the markdown file
with open("UIDAI_HACKATHON_2026_SUBMISSION.md", "r", encoding="utf-8") as f:
    content = f.read()

# Remove anchor links that cause issues
content = re.sub(r'\[([^\]]+)\]\(#[^\)]+\)', r'\1', content)

# Fix image paths to absolute paths
base_path = os.path.dirname(os.path.abspath(__file__))
content = content.replace("output/charts/", os.path.join(base_path, "output/charts/").replace("\\", "/"))

# Create PDF with styling
pdf = MarkdownPdf(toc_level=2)

# Add custom CSS for tricolor theme
css = """
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #1a1a5e;
    line-height: 1.6;
}
h1 {
    color: #2E3192;
    border-bottom: 3px solid;
    border-image: linear-gradient(90deg, #FF9933, #FFFFFF, #138808) 1;
    padding-bottom: 10px;
}
h2 {
    color: #FF9933;
}
h3 {
    color: #138808;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
}
th {
    background: linear-gradient(90deg, #FF9933, #2E3192);
    color: white;
    padding: 10px;
}
td {
    border: 1px solid #ddd;
    padding: 8px;
}
tr:nth-child(even) {
    background-color: #f9f9f9;
}
code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Consolas', monospace;
}
pre {
    background-color: #1a1a5e;
    color: #00ff00;
    padding: 15px;
    border-radius: 8px;
    overflow-x: auto;
}
blockquote {
    border-left: 4px solid #FF9933;
    background: #fff8f0;
    padding: 10px 20px;
    margin: 15px 0;
}
"""

pdf.add_section(Section(content, toc=True))

# Save PDF
output_path = "UIDAI_HACKATHON_2026_SUBMISSION.pdf"
pdf.save(output_path)

print("PDF Created Successfully!")
print(f"File: {output_path}")
print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
