# 500 Character Chunker Office Script

This Office Script reads the text in cell `A1` of the active worksheet, splits it into lines, and groups the lines into chunks that are **no longer than 500 characters**. Lines are never split. If a line is longer than 500 characters, it becomes its own chunk. Chunks are written to a new worksheet in column A.

## How it works
1. The script reads the text in `A1` and splits it by newline characters.
2. It builds chunks by appending full lines until adding another line would exceed 500 characters.
3. The script creates a new worksheet (named `Chunks`, or `Chunks (1)`, etc. if needed) and writes each chunk to a new row in column A.

## Usage
1. Open the workbook and place your large text in cell `A1` of the active worksheet.
2. Open **Automate** → **New Script** in Excel for the web.
3. Paste the contents of [`splitTextToChunks.ts`](./splitTextToChunks.ts) into the script editor.
4. Save and run the script. The chunked output will appear in a new worksheet in column A.
