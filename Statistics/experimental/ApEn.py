import os
from pypdf import PdfWriter

# Directory containing PDF files
directory = r"/Users/ralf/Downloads"    

# Initialize PdfMerger
merger = PdfWriter()

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".pdf"):  # Check if it's a PDF file
        file_path = os.path.join(directory, filename)
        print(f"Adding: {file_path}")
        merger.append(file_path)

# Output file name
output_path = os.path.join(directory, "finalFile.pdf")

# Write the merged PDF to the output file
merger.write(output_path)
merger.close()

print(f"PDF files merged successfully into {output_path}")
