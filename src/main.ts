import './style.css'
import init from './entry';
import { assert } from './utils/util';

async function initializeWebGPU() {
  if (navigator.gpu === undefined) {
    const h = document.querySelector('#title') as HTMLElement;
    h.innerText = 'WebGPU is not supported in this browser.';
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) {
    const h = document.querySelector('#title') as HTMLElement;
    h.innerText = 'No adapter is available for WebGPU.';
    return;
  }
  const device = await adapter.requestDevice();

  const canvas = document.querySelector<HTMLCanvasElement>('#webgpu-canvas');
  assert(canvas !== null);
  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  
  init(context, device); 
}

const dropArea = document.querySelector('#drop-area') as HTMLElement;
assert(dropArea !== null);

dropArea.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropArea.classList.add('dragover');
});

dropArea.addEventListener('dragleave', () => {
  dropArea.classList.remove('dragover');
});

dropArea.addEventListener('drop', async (event) => {
  event.preventDefault();
  dropArea.classList.remove('dragover');

      console.log("Ran webgpu")
  const files = event.dataTransfer?.files;
  if (files && files.length > 0) {
    const file = files[0];
    if (file.type === 'text/csv') {
      const text = await file.text();
      const rows = text.split('\n').map(row => row.split(','));
      initializeWebGPU();
    } else {
      console.error('Please drop a CSV file.');
    }
  }
});
