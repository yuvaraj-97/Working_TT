/**
 * Office Script: Split the text in cell A1 of the active worksheet into
 * chunks of up to 500 characters and write them to a new worksheet.
 */
function main(workbook: ExcelScript.Workbook) {
  const maxLength = 500;
  const sourceSheet = workbook.getActiveWorksheet();
  const sourceText = sourceSheet.getRange("A1").getText() ?? "";

  const lines = sourceText.split(/\r?\n/);
  const chunks: string[] = [];
  let currentChunk = "";

  for (const line of lines) {
    const lineLength = line.length;

    if (lineLength > maxLength) {
      if (currentChunk.length > 0) {
        chunks.push(currentChunk);
        currentChunk = "";
      }
      chunks.push(line);
      continue;
    }

    if (currentChunk.length === 0) {
      currentChunk = line;
      continue;
    }

    const potentialLength = currentChunk.length + 1 + lineLength; // +1 for the newline
    if (potentialLength > maxLength) {
      chunks.push(currentChunk);
      currentChunk = line;
    } else {
      currentChunk += "\n" + line;
    }
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk);
  }

  const outputSheet = workbook.addWorksheet(getUniqueSheetName(workbook, "Chunks"));
  if (chunks.length === 0) {
    outputSheet.getRange("A1").setValue("");
    return;
  }

  const outputValues = chunks.map((chunk) => [chunk]);
  outputSheet.getRangeByIndexes(0, 0, outputValues.length, 1).setValues(outputValues);
}

/**
 * Generates a worksheet name that does not conflict with existing names.
 */
function getUniqueSheetName(workbook: ExcelScript.Workbook, baseName: string): string {
  const existingNames = new Set(workbook.getWorksheets().map((sheet) => sheet.getName()));
  if (!existingNames.has(baseName)) {
    return baseName;
  }

  let suffix = 1;
  let candidate = `${baseName} (${suffix})`;
  while (existingNames.has(candidate)) {
    suffix += 1;
    candidate = `${baseName} (${suffix})`;
  }

  return candidate;
}
