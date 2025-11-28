/**
 * Office Script: Split the text in cell A1 of the active worksheet into
 * chunks of up to 500 characters and overwrite the worksheet with the
 * chunks stored in a single-column table.
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

  const outputSheet = sourceSheet;
  outputSheet.getUsedRange()?.clear(ExcelScript.ClearApplyTo.all);

  const tableValues: string[][] = [["Chunk"]];
  for (const chunk of chunks) {
    tableValues.push([chunk]);
  }

  const tableRange = outputSheet.getRangeByIndexes(0, 0, tableValues.length, 1);
  tableRange.setValues(tableValues);
  outputSheet.addTable(tableRange.getAddress(), true);
}
