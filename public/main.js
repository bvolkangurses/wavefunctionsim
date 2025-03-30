/*
  main.js

  This client-side script does the following:
   1) Sets up mouse-driven potential drawing on potentialCanvas.
   2) Initializes a 2D wavefunction (e.g., a Gaussian).
   3) Solves the time-dependent Schrödinger equation (in a simplified form) frame-by-frame.
   4) Renders the wavefunction's probability density on waveCanvas in real time.
*/

const socket = io(); // Connect to the server (if you want server messages)

// === DOM Elements ===
const potentialCanvas = document.getElementById('potentialCanvas');
const waveCanvas      = document.getElementById('waveCanvas');
const startSimBtn     = document.getElementById('startSimBtn');
const resetSimBtn     = document.getElementById('resetSimBtn');

// Canvas contexts
const potCtx  = potentialCanvas.getContext('2d');
const waveCtx = waveCanvas.getContext('2d');

// Canvas dimensions
const WIDTH  = potentialCanvas.width;
const HEIGHT = potentialCanvas.height;

// Discretize the 2D space
const NX = 100;  // number of discrete points in x (reduced from 200 for performance)
const NY = 100;  // number of discrete points in y
const TOTAL_POINTS = NX * NY;
const potential = new Float32Array(TOTAL_POINTS); // flattened 2D potential array
let isDrawing = false;                  // are we currently drawing with the mouse?

// Utility: convert 2D coordinates to 1D array index
function coordToIndex(x, y) {
  return Math.max(0, Math.min(NX - 1, Math.floor(x))) + 
         Math.max(0, Math.min(NY - 1, Math.floor(y))) * NX;
}

// Utility: map canvas x-coord to grid x-coord
function canvasToGridX(x) {
  return (x / WIDTH) * NX;
}

// Utility: map canvas y-coord to grid y-coord
function canvasToGridY(y) {
  return (y / HEIGHT) * NY;
}

// Utility: map canvas y-coord to a potential value
function yToPotential(y) {
  const maxPotential = 2000.0;
  const invertedY = HEIGHT - y;
  return (invertedY / HEIGHT) * maxPotential;
}

// Redraw the potential surface
function redrawPotential() {
  potCtx.clearRect(0, 0, WIDTH, HEIGHT);

  // Draw the potential as a heatmap
  const imageData = potCtx.createImageData(WIDTH, HEIGHT);
  const data = imageData.data;
  
  for (let canvasY = 0; canvasY < HEIGHT; canvasY++) {
    for (let canvasX = 0; canvasX < WIDTH; canvasX++) {
      // Map canvas coordinates to grid coordinates
      const gridX = canvasToGridX(canvasX);
      const gridY = canvasToGridY(canvasY);
      
      // Get the potential at this point
      const idx = coordToIndex(gridX, gridY);
      const V = potential[idx];
      
      // Convert potential to color (green with varying intensity)
      const intensity = Math.min(255, V * 255 / 2000.0);
      
      // Set pixel data (RGBA)
      const pixelIndex = (canvasY * WIDTH + canvasX) * 4;
      data[pixelIndex] = 0;                 // R
      data[pixelIndex + 1] = intensity;     // G
      data[pixelIndex + 2] = 0;             // B
      data[pixelIndex + 3] = 255;           // A (fully opaque)
    }
  }
  
  potCtx.putImageData(imageData, 0, 0);
}

// Mouse handlers for drawing the potential
potentialCanvas.addEventListener('mousedown', (e) => {
  isDrawing = true;
  drawPotentialAtMouse(e);
});

potentialCanvas.addEventListener('mousemove', (e) => {
  if (isDrawing) {
    drawPotentialAtMouse(e);
  }
});

potentialCanvas.addEventListener('mouseup', () => {
  isDrawing = false;
});

function drawPotentialAtMouse(e) {
  const rect = potentialCanvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  
  // Draw a potential "brush" (multiple points in a small radius)
  const brushSize = 5;
  const potValue = yToPotential(mouseY);
  
  for (let dy = -brushSize; dy <= brushSize; dy++) {
    for (let dx = -brushSize; dx <= brushSize; dx++) {
      const dist = Math.sqrt(dx*dx + dy*dy);
      if (dist <= brushSize) {
        const x = mouseX + dx;
        const y = mouseY + dy;
        if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
          const gridX = canvasToGridX(x);
          const gridY = canvasToGridY(y);
          const idx = coordToIndex(gridX, gridY);
          potential[idx] = potValue;
        }
      }
    }
  }
  
  redrawPotential();
}

// --- Wavefunction Setup and Solver ---
let psiRe = new Float32Array(TOTAL_POINTS);
let psiIm = new Float32Array(TOTAL_POINTS);

// Initialize wavefunction as a 2D Gaussian wave packet
function initWavefunction() {
  const x0 = NX / 4;       // center of the packet (x)
  const y0 = NY / 2;       // center of the packet (y)
  const sigma = NX / 15;   // width of the packet
  const k0x = 1.0;         // initial momentum in x direction
  const k0y = 0.0;         // initial momentum in y direction

  for (let y = 0; y < NY; y++) {
    for (let x = 0; x < NX; x++) {
      const idx = coordToIndex(x, y);
      const r2 = ((x - x0) ** 2 + (y - y0) ** 2);
      const gauss = Math.exp(-0.5 * r2 / (sigma ** 2));
      
      // add a plane wave factor e^{i (k0x * x + k0y * y)}
      const phase = k0x * x + k0y * y;
      psiRe[idx] = gauss * Math.cos(phase);
      psiIm[idx] = gauss * Math.sin(phase);
    }
  }
  normalizeWavefunction();
}

// Normalize the wavefunction
function normalizeWavefunction() {
  let sum = 0;
  for (let i = 0; i < TOTAL_POINTS; i++) {
    sum += psiRe[i] * psiRe[i] + psiIm[i] * psiIm[i];
  }
  const norm = Math.sqrt(sum);
  if (norm > 0) {  // Avoid division by zero
    for (let i = 0; i < TOTAL_POINTS; i++) {
      psiRe[i] /= norm;
      psiIm[i] /= norm;
    }
  }
}

// Time evolution step for 2D Schrodinger equation
function stepWavefunction(dt) {
  /*
   A simplistic Euler step for 2D:
     i dΨ/dt = -1/2 (∇²Ψ) + V Ψ
   where ∇² is now the 2D Laplacian
  */
  const psiReOld = new Float32Array(psiRe);
  const psiImOld = new Float32Array(psiIm);

  const c = 0.5;  // factor from (1/2) in front of the Laplacian

  for (let y = 0; y < NY; y++) {
    for (let x = 0; x < NX; x++) {
      const idx = coordToIndex(x, y);
      
      // Get indices of neighboring points (with boundary conditions)
      const xm = Math.max(0, x - 1);
      const xp = Math.min(NX - 1, x + 1);
      const ym = Math.max(0, y - 1);
      const yp = Math.min(NY - 1, y + 1);
      
      const idxXm = coordToIndex(xm, y);
      const idxXp = coordToIndex(xp, y);
      const idxYm = coordToIndex(x, ym);
      const idxYp = coordToIndex(x, yp);
      
      // 2D Laplacian (5-point stencil)
      const lapRe = psiReOld[idxXm] + psiReOld[idxXp] + 
                   psiReOld[idxYm] + psiReOld[idxYp] - 
                   4.0 * psiReOld[idx];
                   
      const lapIm = psiImOld[idxXm] + psiImOld[idxXp] + 
                   psiImOld[idxYm] + psiImOld[idxYp] - 
                   4.0 * psiImOld[idx];

      // Same calculation as 1D but with 2D Laplacian
      const realPart = -c * lapIm + potential[idx] * psiImOld[idx];
      const imagPart =  c * lapRe - potential[idx] * psiReOld[idx];

      // scale by dt
      psiRe[idx] += dt * realPart;
      psiIm[idx] += dt * imagPart;
    }
  }
}

// Render probability density on waveCanvas as a heatmap
function drawWavefunction() {
  waveCtx.clearRect(0, 0, WIDTH, HEIGHT);
  
  // Create an image for the probability density
  const imageData = waveCtx.createImageData(WIDTH, HEIGHT);
  const data = imageData.data;
  
  // Find the maximum probability for proper normalization
  let maxProb = 0;
  for (let i = 0; i < TOTAL_POINTS; i++) {
    const prob = psiRe[i] * psiRe[i] + psiIm[i] * psiIm[i];
    maxProb = Math.max(maxProb, prob);
  }
  
  maxProb = Math.max(maxProb, 1e-10); // Avoid division by zero
  
  for (let canvasY = 0; canvasY < HEIGHT; canvasY++) {
    for (let canvasX = 0; canvasX < WIDTH; canvasX++) {
      // Map canvas coordinates to grid
      const gridX = canvasToGridX(canvasX);
      const gridY = canvasToGridY(canvasY);
      const idx = coordToIndex(gridX, gridY);
      
      // Calculate probability at this point
      const prob = psiRe[idx] * psiRe[idx] + psiIm[idx] * psiIm[idx];
      
      // Scale probability for visualization
      const intensity = Math.min(255, (prob / maxProb) * 255);
      
      // Set pixel color (yellow)
      const pixelIndex = (canvasY * WIDTH + canvasX) * 4;
      data[pixelIndex] = intensity;        // R
      data[pixelIndex + 1] = intensity;    // G
      data[pixelIndex + 2] = 0;            // B
      data[pixelIndex + 3] = 255;          // A
    }
  }
  
  waveCtx.putImageData(imageData, 0, 0);
}

// Animation loop
let animId = null;
let lastTimestamp = null;

function animate(timestamp) {
  if (!lastTimestamp) lastTimestamp = timestamp;
  const dtSec = (timestamp - lastTimestamp) / 1000;
  lastTimestamp = timestamp;

  // Use multiple small substeps to keep stable
  const nSubSteps = 5; // Reduced from 10 for performance in 2D
  const dt = 0.00025;  // Reduced for stability in 2D
  for (let i = 0; i < nSubSteps; i++) {
    stepWavefunction(dt);
  }

  drawWavefunction();
  animId = requestAnimationFrame(animate);
}

// === Button Handlers ===
startSimBtn.addEventListener('click', () => {
  initWavefunction();
  if (!animId) {
    animId = requestAnimationFrame(animate);
  }
});

resetSimBtn.addEventListener('click', () => {
  // Stop animation
  if (animId) {
    cancelAnimationFrame(animId);
    animId = null;
  }
  // Clear wavefunction and potential
  psiRe.fill(0);
  psiIm.fill(0);
  potential.fill(0);

  // Clear canvases
  potCtx.clearRect(0, 0, WIDTH, HEIGHT);
  waveCtx.clearRect(0, 0, WIDTH, HEIGHT);
});

// Draw initial zero potential
redrawPotential();
