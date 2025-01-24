// import Papa from 'papaparse';

// function handleFile(file: File) {
//   const reader = new FileReader();
//   reader.onload = (event) => {
//     const data = event.target?.result;
//     if (file.type === 'text/csv') {
//       Papa.parse(data as string, {
//         complete: (results) => {
//           console.log('CSV Data:', results.data);
//           // Perform your analysis on CSV data here
//         },
//       });
//     } else {
//       const workbook = XLSX.read(data, { type: 'binary' });
//       const sheetName = workbook.SheetNames[0];
//       const sheet = workbook.Sheets[sheetName];
//       const jsonData = XLSX.utils.sheet_to_json(sheet);
//       console.log('Excel Data:', jsonData);
//       // Perform your analysis on Excel data here
//     }
//   };
//   reader.readAsBinaryString(file);
// }

// function setupDragAndDrop() {
//   const dropArea = document.getElementById('drop-area');
//   dropArea?.addEventListener('dragover', (event) => {
//     event.preventDefault();
//     dropArea.classList.add('dragging');
//   });

//   dropArea?.addEventListener('dragleave', () => {
//     dropArea.classList.remove('dragging');
//   });

//   dropArea?.addEventListener('drop', (event) => {
//     event.preventDefault();
//     dropArea.classList.remove('dragging');
//     const files = event.dataTransfer?.files;
//     if (files && files.length > 0) {
//       handleFile(files[0]);
//     }
//   });
// }

// document.addEventListener('DOMContentLoaded', setupDragAndDrop);