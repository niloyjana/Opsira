/* ── Opsira — Problem 1 & 2 Dashboard Logic ── */

const API_URL = "http://localhost:5000";
let charts = {};

document.addEventListener('DOMContentLoaded', () => {
  setupUI();
  // Check health
  fetch(`${API_URL}/health`).then(r => r.json()).then(d => console.log("Opsira Backend:", d.status));
});

function setupUI() {
  const fileInput = document.getElementById('fileInput');
  const uploadZone = document.getElementById('uploadZone');

  uploadZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) processFile(e.target.files[0]);
  });

  // Drag & Drop
  uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) processFile(e.dataTransfer.files[0]);
  });
}

async function processFile(file) {
  if (!file.type.startsWith('image/')) return;

  showLoading(true);
  
  const formData = new FormData();
  formData.append('image', file); // API expects key "image"

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    renderDashboard(data);
    renderLowerDeck(data);
  } catch (err) {
    console.error("Analysis Failed:", err);
    alert("Analysis Failed: Check backend connection.");
  } finally {
    showLoading(false);
  }
}

function renderDashboard(data) {
  // 1. Diagnosis Information
  document.getElementById('diagTitle').textContent = data.label;
  document.getElementById('diagEmoji').textContent = getEmoji(data.prediction);
  document.getElementById('confValue').textContent = `${data.confidence}% Confidence`;
  document.getElementById('confBar').style.width = `${data.confidence}%`;
  
  const riskBadge = document.getElementById('riskBadge');
  riskBadge.textContent = data.severity;
  riskBadge.className = `risk-badge risk-${data.severity.toLowerCase()}`;
  
  document.getElementById('adviceBox').textContent = data.clinical_advice;
  document.getElementById('urgencyBox').innerHTML = `<strong>PROTOCOL:</strong> ${data.protocol}`;

  // 2. Images (Original & Heatmap)
  document.getElementById('previewImg').src = `data:image/jpeg;base64,${data.original_image}`;
  document.getElementById('heatmapImg').src = `data:image/jpeg;base64,${data.heatmap_image}`;

  // 3. Metric Bars (Problem 1 Requirement)
  updateMetricBars(data.metrics);

  // 4. Probability Bars (Problem 2 Requirement)
  updateProbabilityBars(data.probabilities, data.prediction);

  // 5. Radar Chart
  updateRadarChart(data.radar);

  // Show results panel
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('results').style.display = 'grid';
}

function updateMetricBars(metrics) {
  const container = document.getElementById('deviationList');
  container.innerHTML = ""; // Clear existing

  Object.entries(metrics).forEach(([key, val]) => {
    const severity = val > 75 ? 'critical' : val > 40 ? 'warning' : 'normal';
    const color = severity === 'critical' ? 'var(--red)' : severity === 'warning' ? 'var(--amber)' : 'var(--green)';
    
    container.innerHTML += `
      <div class="dev-item">
        <div class="dev-header">
          <span style="color:var(--text-dim); text-transform: uppercase; font-size: 0.6rem; font-weight: 700;">${key}</span>
          <span style="color:${color}">${val}%</span>
        </div>
        <div class="dev-bar-bg">
          <div class="dev-bar-fill" style="width:${val}%; background:${color}"></div>
        </div>
      </div>
    `;
  });
}

function updateProbabilityBars(probs, prediction) {
  // Note: Assuming there is a container for probability bars or they are part of Chart.js
  // Let's also update the bar chart via Chart.js
  const ctx = document.getElementById('probChart').getContext('2d');
  
  if (charts.prob) charts.prob.destroy();

  const labels = Object.keys(probs).map(s => s.replace('_', ' ').toUpperCase());
  const values = Object.values(probs);

  charts.prob = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Probability %',
        data: values,
        backgroundColor: Object.keys(probs).map(k => k === prediction ? '#00f0ff' : 'rgba(240, 246, 252, 0.1)'),
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255,255,255,0.05)' } },
        x: { grid: { display: false } }
      }
    }
  });
}

function updateRadarChart(radar) {
  const ctx = document.getElementById('featChart').getContext('2d');
  
  if (charts.radar) charts.radar.destroy();

  charts.radar = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: radar.labels.map(l => l.toUpperCase()),
      datasets: [
        {
          label: 'Current Scan',
          data: radar.current,
          borderColor: '#00f0ff',
          backgroundColor: 'rgba(0, 240, 255, 0.1)',
          pointBackgroundColor: '#00f0ff',
          fill: true
        },
        {
          label: 'Healthy Baseline',
          data: radar.baseline,
          borderColor: 'rgba(0, 255, 149, 0.3)',
          borderDash: [5, 5],
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0, max: 100,
          angleLines: { color: 'rgba(255,255,255,0.05)' },
          grid: { color: 'rgba(255,255,255,0.05)' },
          pointLabels: { color: '#8b949e', font: { size: 10 } },
          ticks: { display: false }
        }
      },
      plugins: { legend: { labels: { color: '#f0f6fc', boxWidth: 10 } } }
    }
  });
}

function showLoading(show) {
  const overlay = document.getElementById('loadingOverlay');
  const skeleton = document.getElementById('skeletonGrid');
  const empty = document.getElementById('emptyState');
  const results = document.getElementById('results');
  const lower = document.getElementById('lowerDeck');

  if (show) {
    overlay.style.display = 'flex';
    skeleton.style.display = 'grid';
    empty.style.display = 'none';
    results.style.display = 'none';
    lower.style.display = 'none';
  } else {
    overlay.style.display = 'none';
    skeleton.style.display = 'none';
  }
}

function renderLowerDeck(data) {
  const lower = document.getElementById('lowerDeck');
  const recoGrid = document.getElementById('recoGrid');
  const dietChart = document.getElementById('dietChart');
  
  lower.style.display = 'flex';
  
  // Recommendations
  const recos = getRecommendations(data.prediction);
  recoGrid.innerHTML = recos.map(r => `
    <div class="reco-item">
      <h4>${r.title}</h4>
      <p>${r.text}</p>
    </div>
  `).join('');
  
  // Diet Chart
  const diet = getDiet(data.prediction);
  dietChart.innerHTML = diet.map(d => `
    <div class="diet-card">
      <span class="icon">${d.icon}</span>
      <label>${d.label}</label>
      <p>${d.value}</p>
    </div>
  `).join('');

  setupScrollReveal();
}

function getRecommendations(pred) {
  const base = [
    { title: "Standard Protocol", text: "Ensure 20-20-20 rule during screen time." },
    { title: "Hydration", text: "Maintain optimal fluid intake for tear film stability." }
  ];
  const specific = {
    cataract: [
      { title: "UV Protection", text: "Wear polarized sunglasses to prevent further protein cross-linking." },
      { title: "Surgical Prep", text: "Consult with a surgeon about intraocular lens (IOL) options." }
    ],
    jaundice: [
      { title: "Liver Panel", text: "Request an immediate metabolic panel (ALT/AST levels)." },
      { title: "Alcohol Cessation", text: "Zero tolerance for hepatic stressors during recovery." }
    ],
    red_eye: [
      { title: "Compresse", text: "Apply cool compresses to reduce vascular dilation." },
      { title: "Contagion Control", text: "Avoid sharing towels or touching eyes to prevent spread." }
    ],
    healthy: [
      { title: "Routine Check", text: "Annual comprehensive dilated eye exams are recommended." },
      { title: "Lutein Intake", text: "Continue high intake of leafy greens to support macular pigment." }
    ]
  };
  return [...(specific[pred] || []), ...base];
}

function getDiet(pred) {
  const diet = [
    { icon: "🥕", label: "Beta-Carotene", value: "Carrots, Sweet Potatoes" },
    { icon: "🐟", label: "Omega-3", value: "Salmon, Flaxseeds" },
    { icon: "🥗", label: "Lutein", value: "Spinach, Kale" }
  ];
  if (pred === 'jaundice') {
    return [
      { icon: "🍋", label: "Citrus", value: "Lemon water to flush toxins" },
      { icon: "🍵", label: "Dandelion", value: "Liver supporting herbal tea" },
      { icon: "🍚", label: "Whole Grains", value: "Easily digestible fiber" }
    ];
  }
  return diet;
}

function setupScrollReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.reveal-card').forEach(card => {
    card.classList.remove('visible');
    observer.observe(card);
  });
}

function getEmoji(pred) {
  const emojis = { cataract: "🔵", healthy: "✅", jaundice: "🟡", red_eye: "🔴" };
  return emojis[pred] || "🔍";
}

function resetApp() {
  document.getElementById('results').style.display = 'none';
  document.getElementById('emptyState').style.display = 'flex';
  document.getElementById('fileInput').value = "";
}

function toggleWebcam() {
  // Existing webcam logic remains same...
  const wrap = document.getElementById('webcamWrap');
  if (wrap.style.display === 'none') {
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
      document.getElementById('video').srcObject = stream;
      wrap.style.display = 'flex';
    });
  } else {
    const stream = document.getElementById('video').srcObject;
    if (stream) stream.getTracks().forEach(t => t.stop());
    wrap.style.display = 'none';
  }
}

function captureWebcam() {
  const video = document.getElementById('video');
  const canvas = document.getElementById('snapCanvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob(blob => {
    const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
    processFile(file);
    toggleWebcam();
  }, 'image/jpeg');
}
