// Visora prototype script.js
// Requires index.html with video, overlay, buttons and the TF/Tesseract cdn imports.

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const describeBtn = document.getElementById('describeBtn');
const pauseBtn = document.getElementById('pauseBtn');
const sceneBtn = document.getElementById('sceneBtn');
const textBtn = document.getElementById('textBtn');
const outputText = document.getElementById('outputText');
const voiceSelect = document.getElementById('voiceSelect');
const rateRange = document.getElementById('rateRange');
const rateVal = document.getElementById('rateVal');

let model = null;           // COCO-SSD model
let running = false;        // detection loop
let mode = 'scene';         // 'scene' or 'text'
let detectInterval = null;
let lastSpoken = '';
let voices = [];

// --- Setup camera ---
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    video.srcObject = stream;
    await video.play();
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
  } catch (e) {
    console.error('Camera error', e);
    output('Camera access denied or not available. Use secure context (localhost or https).');
  }
}

function resizeCanvas() {
  overlay.width = video.videoWidth || video.clientWidth;
  overlay.height = video.videoHeight || video.clientHeight;
}

// --- Load Model ---
async function loadModel() {
  output('Loading object detection model (COCO-SSD)...');
  model = await cocoSsd.load();
  output('Model loaded. Ready.');
}

// --- Output helper ---
function output(text) {
  outputText.innerText = text;
  console.log('Visora:', text);
}

// --- Speak helper ---
function speak(text, options = {}) {
  if (!('speechSynthesis' in window)) {
    console.warn('SpeechSynthesis not supported');
    return;
  }
  // avoid re-reading same content repeatedly
  if (text.trim() === lastSpoken) return;
  lastSpoken = text;

  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = parseFloat(rateRange.value) || 1;
  const selected = voiceSelect.value;
  if (selected) {
    const v = voices.find(x => x.name === selected);
    if (v) utter.voice = v;
  }
  // small delay to keep UX smooth
  setTimeout(() => speechSynthesis.speak(utter), 100);
}

// --- Load voices into select ---
function populateVoices() {
  voices = speechSynthesis.getVoices().filter(v => v.lang.startsWith('en') || v.lang.startsWith('hi') || v.lang.startsWith('en-'));
  // fallback: if empty, use all voices
  if (!voices.length) voices = speechSynthesis.getVoices();
  voiceSelect.innerHTML = '';
  voices.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v.name;
    opt.textContent = `${v.name} — ${v.lang}${v.default ? ' (default)' : ''}`;
    voiceSelect.appendChild(opt);
  });
}
// Some browsers fire voiceschanged later
speechSynthesis.onvoiceschanged = populateVoices;
populateVoices();

// update rate display
rateRange.addEventListener('input', () => rateVal.textContent = rateRange.value + 'x');

// --- Scene detection loop ---
async function detectLoop() {
  if (!model || video.paused || video.ended) return;
  try {
    const predictions = await model.detect(video);
    drawPredictions(predictions);
    if (predictions.length === 0) {
      output('No objects confidently detected.');
      // optionally speak less often
    } else {
      const top = summarizePredictions(predictions);
      output(top.text);
      speak(top.speech);
    }
  } catch (err) {
    console.error('Detect error', err);
  }
}

// Summarize top objects into a human sentence
function summarizePredictions(preds) {
  // Filter by score threshold
  const threshold = 0.55;
  const filtered = preds.filter(p => p.score >= threshold);
  // Sort by score desc
  filtered.sort((a,b) => b.score - a.score);
  const names = filtered.slice(0,6).map(p => p.class);
  const unique = [...new Set(names)];
  if (unique.length === 0) {
    return { text: 'No confident objects', speech: 'I do not see anything recognizable.' };
  }
  // Build sentence
  const shortList = unique.slice(0,3);
  let text = 'I see ' + shortList.join(', ');
  let speech = '';
  if (shortList.length === 1) {
    speech = `I see a ${shortList[0]}.`;
  } else if (shortList.length === 2) {
    speech = `I see a ${shortList[0]} and a ${shortList[1]}.`;
  } else {
    speech = `I see a ${shortList[0]}, a ${shortList[1]}, and a ${shortList[2]}.`;
  }
  // maybe add count for repeated items
  return { text, speech };
}

// Draw bounding boxes for visual feedback
function drawPredictions(predictions) {
  const w = overlay.width, h = overlay.height;
  ctx.clearRect(0,0,w,h);
  // semi-transparent backdrop for readability
  ctx.fillStyle = 'rgba(0,0,0,0.06)';
  ctx.fillRect(0,0,w,h);

  predictions.forEach(pred => {
    if (pred.score < 0.5) return;
    const [x,y,width,height] = pred.bbox;
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = Math.max(2, (pred.score * 4));
    ctx.strokeRect(x, y, width, height);
    // label background
    ctx.fillStyle = 'rgba(6,11,22,0.7)';
    ctx.fillRect(x, y - 20, ctx.measureText(pred.class).width + 10, 20);
    ctx.fillStyle = '#e6eef8';
    ctx.fillText(`${pred.class} (${Math.round(pred.score * 100)}%)`, x + 6, y - 6);
  });
}

// --- OCR (Tesseract) ---
async function runOCR() {
  output('Capturing frame for OCR...');
  // capture current video frame
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const cctx = canvas.getContext('2d');
  cctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // optional: crop center area to speed up OCR
  // const cropW = canvas.width * 0.9, cropH = canvas.height * 0.4;
  // const sx = (canvas.width - cropW)/2, sy = (canvas.height - cropH)/2;
  // const cropped = document.createElement('canvas');
  // cropped.width = cropW; cropped.height = cropH;
  // const cc = cropped.getContext('2d');
  // cc.drawImage(canvas, sx, sy, cropW, cropH, 0, 0, cropW, cropH);

  const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

  output('Running OCR, please wait...');
  try {
    const { data: { text } } = await Tesseract.recognize(dataUrl, 'eng', {
      logger: m => {
        // progress logging (optional)
        // console.log(m);
      }
    });
    const cleaned = text.trim();
    if (!cleaned) {
      output('No readable text found.');
      speak('I could not read any text in this frame.');
      return;
    }
    output(cleaned);
    speak(cleaned);
  } catch (err) {
    console.error('OCR error', err);
    output('OCR failed. Try better lighting or move closer.');
    speak('I could not read that. Try a clearer image or brighter light.');
  }
}

// --- Controls ---
sceneBtn.addEventListener('click', () => {
  mode = 'scene';
  sceneBtn.classList.add('active');
  textBtn.classList.remove('active');
  output('Mode: Scene');
});

textBtn.addEventListener('click', () => {
  mode = 'text';
  textBtn.classList.add('active');
  sceneBtn.classList.remove('active');
  output('Mode: Text (OCR)');
});

describeBtn.addEventListener('click', async () => {
  if (mode === 'scene') {
    // single-run detection and speak
    if (!model) {
      output('Model loading — please wait...');
      return;
    }
    const preds = await model.detect(video);
    drawPredictions(preds);
    const summary = summarizePredictions(preds);
    output(summary.text);
    speak(summary.speech);
  } else {
    // OCR mode
    await runOCR();
  }
});

// Pause/resume detection loop
pauseBtn.addEventListener('click', () => {
  running = !running;
  if (!running) {
    clearInterval(detectInterval);
    pauseBtn.textContent = '▶ Resume';
    output('Detection paused.');
  } else {
    startAutoDetect();
    pauseBtn.textContent = '⏸ Pause';
    output('Detection resumed.');
  }
});

// Auto-detect every N ms when running
function startAutoDetect() {
  if (!model) return;
  if (detectInterval) clearInterval(detectInterval);
  detectInterval = setInterval(async () => {
    if (mode !== 'scene') return;
    await detectLoop();
  }, 1400); // ~1.4s between detections
  running = true;
  pauseBtn.textContent = '⏸ Pause';
}

// initialize on page load
(async function init() {
  output('Starting Visora...');
  await startCamera();
  await loadModel();
  populateVoices();
  // small styling for canvas text
  ctx.font = '16px Inter, Arial';
  ctx.fillStyle = '#e6eef8';
  ctx.textBaseline = 'top';

  // start periodic auto detection by default (scene)
  startAutoDetect();
})();

// Safety note: getUserMedia needs secure context (localhost or https).
