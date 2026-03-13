/* =========================================================
   MNIST Digit Recognition — main.js
   TensorFlow.js CNN trained in-browser
   ========================================================= */

"use strict";

// ── State ──────────────────────────────────────────────────
let model        = null;
let mnistData    = null;
let isTrained    = false;
let isTraining   = false;
let brushRadius  = 10;
let isDrawing    = false;

const EPOCHS       = 10;
const BATCH_SIZE   = 128;
const STEPS_EPOCH  = 150;   // mini-batches per epoch

// ── DOM Refs ───────────────────────────────────────────────
const statusDot    = document.getElementById('status-dot');
const statusText   = document.getElementById('status-text');
const logEl        = document.getElementById('log');
const progressBar  = document.getElementById('progress-bar');
const statEpoch    = document.getElementById('stat-epoch');
const statLoss     = document.getElementById('stat-loss');
const statAcc      = document.getElementById('stat-acc');
const statTest     = document.getElementById('stat-test');
const drawCanvas   = document.getElementById('drawCanvas');
const ctx          = drawCanvas.getContext('2d');
const previewCanvas= document.getElementById('preview');
const predDigit    = document.getElementById('prediction-digit');
const confText     = document.getElementById('confidence-text');
const confBar      = document.getElementById('confidence-bar');
const probBarsEl   = document.getElementById('prob-bars');
const galleryEl    = document.getElementById('mnist-gallery');

// ── Initial canvas fill ────────────────────────────────────
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, 280, 280);
initProbBars();

// ── Charts ─────────────────────────────────────────────────
const chartOptions = (label, color) => ({
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label,
      data: [],
      borderColor: color,
      backgroundColor: color + '22',
      borderWidth: 2,
      pointRadius: 3,
      pointBackgroundColor: color,
      tension: 0.35,
      fill: true,
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: { legend: { display: false } },
    scales: {
      x: {
        ticks: { color: '#4a5568', font: { size: 10 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
      },
      y: {
        ticks: { color: '#4a5568', font: { size: 10 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
      }
    }
  }
});

const lossChart = new Chart(
  document.getElementById('lossChart').getContext('2d'),
  chartOptions('Loss', '#4fd1c5')
);
const accChart = new Chart(
  document.getElementById('accChart').getContext('2d'),
  chartOptions('Accuracy', '#9f7aea')
);

// ── Logging ────────────────────────────────────────────────
function log(msg, cls = '') {
  const line = document.createElement('div');
  line.innerHTML = msg;
  if (cls) line.className = cls;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(msg, mode = '') {
  statusText.textContent = msg;
  statusDot.className = mode;
}

// ── Model Architecture ─────────────────────────────────────
function buildModel() {
  const m = tf.sequential();

  m.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 32,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  m.add(tf.layers.batchNormalization());
  m.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  m.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  m.add(tf.layers.batchNormalization());
  m.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  m.add(tf.layers.dropout({ rate: 0.3 }));
  m.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  m.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return m;
}

// ── Training ───────────────────────────────────────────────
async function startTraining() {
  if (isTraining) return;
  isTraining = true;

  document.getElementById('btn-train').disabled   = true;
  document.getElementById('btn-test').disabled    = true;
  document.getElementById('btn-samples').disabled = true;

  // Reset charts
  lossChart.data.labels = [];
  lossChart.data.datasets[0].data = [];
  lossChart.update();
  accChart.data.labels = [];
  accChart.data.datasets[0].data = [];
  accChart.update();

  logEl.innerHTML = '';
  isTrained = false;

  try {
    // Load dataset
    setStatus('Loading MNIST dataset…', 'active');
    log('⟳ Loading MNIST dataset (55,000 images)…', 'log-info');
    mnistData = new MnistData();
    await mnistData.load();
    log('✔ Dataset loaded successfully.', 'log-success');

    // Build model
    setStatus('Building CNN model…', 'active');
    log('⟳ Building CNN model…', 'log-info');
    model = buildModel();
    const pCount = model.countParams();
    log(`✔ Model ready — ${pCount.toLocaleString()} trainable parameters.`, 'log-success');

    // Training loop
    setStatus('Training…', 'active');
    log('⟳ Starting training for 10 epochs…', 'log-info');

    const totalSteps = EPOCHS * STEPS_EPOCH;
    let stepsDone = 0;

    for (let epoch = 0; epoch < EPOCHS; epoch++) {
      let epochLoss = 0;
      let epochAcc  = 0;

      for (let i = 0; i < STEPS_EPOCH; i++) {
        const batch = mnistData.nextTrainBatch(BATCH_SIZE);

        const history = await model.fit(
          batch.xs.reshape([BATCH_SIZE, 28, 28, 1]),
          batch.labels,
          { batchSize: BATCH_SIZE, epochs: 1, verbose: 0 }
        );

        epochLoss += history.history.loss[0];
        epochAcc  += history.history.acc[0];

        batch.xs.dispose();
        batch.labels.dispose();

        stepsDone++;
        progressBar.style.width = ((stepsDone / totalSteps) * 100).toFixed(1) + '%';
      }

      const avgLoss = (epochLoss / STEPS_EPOCH).toFixed(4);
      const avgAcc  = ((epochAcc  / STEPS_EPOCH) * 100).toFixed(2);

      // Update stats
      statEpoch.innerHTML = `${epoch + 1}<span style="font-size:.9rem;color:var(--text-muted)">/10</span>`;
      statLoss.textContent  = avgLoss;
      statAcc.textContent   = avgAcc + '%';

      // Update charts
      const label = `E${epoch + 1}`;
      lossChart.data.labels.push(label);
      lossChart.data.datasets[0].data.push(parseFloat(avgLoss));
      lossChart.update();

      accChart.data.labels.push(label);
      accChart.data.datasets[0].data.push(parseFloat(avgAcc));
      accChart.update();

      log(`<span style="color:var(--text-muted)">Epoch ${epoch + 1}/10</span> — loss: <span style="color:#4fd1c5">${avgLoss}</span>  acc: <span style="color:#68d391">${avgAcc}%</span>`, 'log-epoch');

      setStatus(`Training… Epoch ${epoch + 1}/${EPOCHS}`, 'active');
    }

    isTrained = true;
    progressBar.style.width = '100%';
    setStatus('Training complete ✅', 'success');
    log('✔ Training complete!', 'log-success');

    document.getElementById('btn-test').disabled    = false;
    document.getElementById('btn-samples').disabled = false;

  } catch (err) {
    setStatus('Error: ' + err.message, 'error');
    log('✖ Error: ' + err.message);
    console.error(err);
  }

  document.getElementById('btn-train').disabled = false;
  isTraining = false;
}

// ── Testing ────────────────────────────────────────────────
async function testModel() {
  if (!isTrained) return;
  setStatus('Evaluating on test set…', 'active');
  log('⟳ Running on 10,000 test images…', 'log-info');

  const TEST_BATCH = 1000;
  const BATCHES    = 10;
  let totalAcc = 0;

  for (let i = 0; i < BATCHES; i++) {
    const batch = mnistData.nextTestBatch(TEST_BATCH);
    const res   = model.evaluate(
      batch.xs.reshape([TEST_BATCH, 28, 28, 1]),
      batch.labels,
      { batchSize: TEST_BATCH }
    );
    totalAcc += (await res[1].data())[0];
    batch.xs.dispose();
    batch.labels.dispose();
    tf.dispose(res);
  }

  const finalAcc = ((totalAcc / BATCHES) * 100).toFixed(2);
  statTest.textContent = finalAcc + '%';
  log(`✔ Test accuracy: <span style="color:#9f7aea;font-weight:700">${finalAcc}%</span>`, 'log-success');
  setStatus(`Test accuracy: ${finalAcc}%`, 'success');
}

// ── MNIST Gallery ──────────────────────────────────────────
async function showSamples() {
  if (!isTrained) return;
  galleryEl.innerHTML = '';
  const N = 40;
  const batch = mnistData.nextTestBatch(N);
  const imgs   = await batch.xs.data();
  const labels = await batch.labels.data();

  for (let i = 0; i < N; i++) {
    const trueLabel = Array.from(labels.slice(i * 10, i * 10 + 10)).indexOf(1);
    const imgData   = imgs.slice(i * 784, i * 784 + 784);

    const item = document.createElement('div');
    item.className = 'gallery-item';

    const c = document.createElement('canvas');
    c.width  = 28;
    c.height = 28;
    c.style.width  = '48px';
    c.style.height = '48px';

    const imgArr = new Uint8ClampedArray(784 * 4);
    for (let j = 0; j < 784; j++) {
      const v = Math.round(imgData[j] * 255);
      imgArr[j * 4]     = v;
      imgArr[j * 4 + 1] = v;
      imgArr[j * 4 + 2] = v;
      imgArr[j * 4 + 3] = 255;
    }
    const id = new ImageData(imgArr, 28, 28);
    c.getContext('2d').putImageData(id, 0, 0);

    const lbl = document.createElement('div');
    lbl.className   = 'gallery-label';
    lbl.textContent = trueLabel;

    item.appendChild(c);
    item.appendChild(lbl);
    item.title = `True label: ${trueLabel}`;

    // Click to predict this sample
    item.onclick = () => {
      // Paint onto drawCanvas
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, 280, 280);
      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width  = 28;
      tmpCanvas.height = 28;
      tmpCanvas.getContext('2d').putImageData(id, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tmpCanvas, 0, 0, 280, 280);
      predictDigit();
    };

    galleryEl.appendChild(item);
  }

  batch.xs.dispose();
  batch.labels.dispose();

  log(`✔ Loaded ${N} sample images from test set.`, 'log-success');
}

// ── Reset ──────────────────────────────────────────────────
function resetAll() {
  if (model) { model.dispose(); model = null; }
  isTrained  = false;
  isTraining = false;

  lossChart.data.labels = [];
  lossChart.data.datasets[0].data = [];
  lossChart.update();
  accChart.data.labels = [];
  accChart.data.datasets[0].data = [];
  accChart.update();

  statEpoch.innerHTML = `0<span style="font-size:.9rem;color:var(--text-muted)">/10</span>`;
  statLoss.textContent = '—';
  statAcc.textContent  = '—';
  statTest.textContent = '—';
  progressBar.style.width = '0%';

  logEl.innerHTML = 'Model reset. Ready to train again.';
  galleryEl.innerHTML = '<span style="color:var(--text-muted);font-size:.8rem">Train model first to see samples</span>';

  clearCanvas();
  setStatus('Ready to train. Click "Train Model" to begin.', '');

  document.getElementById('btn-train').disabled   = false;
  document.getElementById('btn-test').disabled    = true;
  document.getElementById('btn-samples').disabled = true;
}

// ── Canvas Drawing ─────────────────────────────────────────
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, 280, 280);

function getPos(e) {
  if (e.touches) {
    const rect = drawCanvas.getBoundingClientRect();
    return {
      x: e.touches[0].clientX - rect.left,
      y: e.touches[0].clientY - rect.top
    };
  }
  return { x: e.offsetX, y: e.offsetY };
}

function drawDot(e) {
  const p = getPos(e);
  ctx.fillStyle = '#fff';
  // Soft brush: circle with gradient
  const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, brushRadius);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.6, 'rgba(255,255,255,0.85)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = grd;
  ctx.beginPath();
  ctx.arc(p.x, p.y, brushRadius, 0, Math.PI * 2);
  ctx.fill();
}

drawCanvas.addEventListener('mousedown',  (e) => { isDrawing = true; drawDot(e); drawCanvas.classList.add('active'); });
drawCanvas.addEventListener('mousemove',  (e) => { if (!isDrawing) return; drawDot(e); });
drawCanvas.addEventListener('mouseup',    ()  => { isDrawing = false; drawCanvas.classList.remove('active'); updatePreview(); if (isTrained) predictDigit(); });
drawCanvas.addEventListener('mouseleave', ()  => { if (isDrawing) { isDrawing = false; drawCanvas.classList.remove('active'); updatePreview(); if (isTrained) predictDigit(); } });

drawCanvas.addEventListener('touchstart', (e) => { e.preventDefault(); isDrawing = true; drawDot(e); }, { passive: false });
drawCanvas.addEventListener('touchmove',  (e) => { e.preventDefault(); if (!isDrawing) return; drawDot(e); }, { passive: false });
drawCanvas.addEventListener('touchend',   (e) => { e.preventDefault(); isDrawing = false; updatePreview(); if (isTrained) predictDigit(); }, { passive: false });

function clearCanvas() {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, 280, 280);
  predDigit.textContent = '?';
  confText.textContent  = 'Draw on the canvas →';
  confText.style.color  = 'var(--text-muted)';
  confText.style.fontSize = '1rem';
  confBar.style.width = '0%';
  resetProbBars();
  updatePreview();
}

function updatePreview() {
  const pCtx = previewCanvas.getContext('2d');
  pCtx.imageSmoothingEnabled = true;
  pCtx.drawImage(drawCanvas, 0, 0, 28, 28);
}

// ── Brush Size ─────────────────────────────────────────────
function setBrush(r) {
  brushRadius = r;
  ['brush-sm','brush-md','brush-lg'].forEach(id => document.getElementById(id).classList.remove('active'));
  if (r === 10) document.getElementById('brush-sm').classList.add('active');
  else if (r === 16) document.getElementById('brush-md').classList.add('active');
  else document.getElementById('brush-lg').classList.add('active');
}

// ── Prediction ─────────────────────────────────────────────
async function predictDigit() {
  if (!model || !isTrained) return;

  // Preprocess: grab 28×28 from preview canvas
  updatePreview();
  const pCtx = previewCanvas.getContext('2d');
  const imgData = pCtx.getImageData(0, 0, 28, 28);
  const pixels  = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    pixels[i] = imgData.data[i * 4] / 255;   // R channel (grayscale)
  }

  const input = tf.tensor4d(pixels, [1, 28, 28, 1]);
  const preds = model.predict(input);
  const data  = await preds.data();
  input.dispose();
  preds.dispose();

  const digit      = data.indexOf(Math.max(...data));
  const confidence = (Math.max(...data) * 100).toFixed(1);

  predDigit.textContent  = digit;
  confText.textContent   = `${confidence}% confident`;
  confText.style.color   = 'var(--accent-green)';
  confText.style.fontSize = '1.3rem';
  confBar.style.width    = confidence + '%';

  updateProbBars(data);
}

// ── Probability Bars ───────────────────────────────────────
function initProbBars() {
  probBarsEl.innerHTML = '';
  for (let i = 0; i < 10; i++) {
    probBarsEl.innerHTML += `
      <div class="prob-row">
        <div class="prob-label">${i}</div>
        <div class="prob-track"><div class="prob-fill" id="prob-fill-${i}"></div></div>
        <div class="prob-pct" id="prob-pct-${i}">0.0%</div>
      </div>`;
  }
}

function updateProbBars(data) {
  const maxIdx = data.indexOf(Math.max(...data));
  for (let i = 0; i < 10; i++) {
    const pct  = (data[i] * 100).toFixed(1);
    const fill = document.getElementById(`prob-fill-${i}`);
    const pctEl= document.getElementById(`prob-pct-${i}`);
    fill.style.width = pct + '%';
    pctEl.textContent = pct + '%';
    fill.className = 'prob-fill' + (i === maxIdx ? ' top' : '');
  }
}

function resetProbBars() {
  for (let i = 0; i < 10; i++) {
    document.getElementById(`prob-fill-${i}`).style.width = '0%';
    document.getElementById(`prob-pct-${i}`).textContent = '0.0%';
    document.getElementById(`prob-fill-${i}`).className = 'prob-fill';
  }
}