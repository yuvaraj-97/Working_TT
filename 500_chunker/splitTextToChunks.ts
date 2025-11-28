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

  const outputSheetName = "chunks";
  const tableName = "chunks";

  let outputSheet = workbook.getWorksheet(outputSheetName);
  if (!outputSheet) {
    outputSheet = workbook.addWorksheet(outputSheetName);
  }

  outputSheet.getUsedRange()?.clear(ExcelScript.ClearApplyTo.all);

  const tableValues: string[][] = [["Chunk"]];
  for (const chunk of chunks) {
    tableValues.push([chunk]);
  }

  const tableRange = outputSheet.getRangeByIndexes(0, 0, tableValues.length, 1);
  tableRange.setValues(tableValues);

  const existingTable = workbook
    .getTables()
    .find((table) => table.getName() === tableName);

  if (existingTable) {
    if (existingTable.getWorksheet().getName() !== outputSheetName) {
      existingTable.delete();
      const newTable = outputSheet.addTable(tableRange.getAddress(), true);
      newTable.setName(tableName);
    } else {
      existingTable.resize(tableRange);
      existingTable.getRange().setValues(tableValues);
    }
  } else {
    const newTable = outputSheet.addTable(tableRange.getAddress(), true);
    newTable.setName(tableName);
  }
}
