"""
Convert Markdown to PDF for UIDAI Hackathon Submission
"""

from markdown_pdf import MarkdownPdf, Section
import re

# Read the markdown file
with open("UIDAI_HACKATHON_2026_SUBMISSION.md", "r", encoding="utf-8") as f:
    content = f.read()

# Remove TOC anchor links that cause issues
content = re.sub(r'\[([^\]]+)\]\(#[^\)]+\)', r'\1', content)

pdf = MarkdownPdf(toc_level=2)

# Add section
pdf.add_section(Section(content))

# Save PDF
pdf.save("UIDAI_HACKATHON_2026_SUBMISSION.pdf")

print("✅ PDF Created Successfully!")
print("📄 File: UIDAI_HACKATHON_2026_SUBMISSION.pdf")
