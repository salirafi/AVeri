const errorEl = document.getElementById("error");
const resultCard = document.getElementById("result-card");
const probabilityRingEl = document.getElementById("probability-ring");
const probabilityEl = document.getElementById("probability");
const classificationConfidenceEl = document.getElementById("classification-confidence");
const classificationDecisionEl = document.getElementById("classification-decision");
const buttonEl = document.getElementById("predict-btn");
const decisionThreshold = 0.58;


// number formatting
function formatNumber(value) {
  return value == null || Number.isNaN(Number(value)) ? "-" : Number(value).toFixed(4);
}


function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden"); // remove hidden CSS class
}
function clearError() { // clear the error
  errorEl.textContent = "";
  errorEl.classList.add("hidden");
}


// function metricRating(value) {
//   const score = Number(value);
//   if (Number.isNaN(score)) return "-";
//   if (score >= 0.9) return "Very strong";
//   if (score >= 0.8) return "Good";
//   if (score >= 0.7) return "Fairly good";
//   if (score >= 0.6) return "Moderate";
//   return "Limited";
// }


function probabilityColor(probability) {
  const clamped = Math.max(0, Math.min(1, Number(probability)));
  const hue = 8 + clamped * 126; // low prob is near red/orange; high prob more green
  return `hsl(${hue} 72% 46%)`;
}

function classificationConfidence(probability) {
  if (probability >= 0.9 || probability <= 0.1) return "Surely";
  if (probability >= 0.75 || probability <= 0.25) return "Very Likely";
  if (probability >= 0.6 || probability <= 0.4) return "Likely";
  return "Uncertain";
}

function renderProbability(probability, showClassification = true) {

  const clamped = Math.max(0, Math.min(1, Number(probability)));
  const angle = `${(clamped * 360).toFixed(2)}deg`;
  const color = probabilityColor(clamped);
  const decision = clamped >= decisionThreshold ? "Same author" : "Different author";
  const decisionClass = clamped >= decisionThreshold ? "is-same" : "is-different";

  probabilityRingEl.style.setProperty("--ring-angle", angle);
  probabilityRingEl.style.setProperty("--ring-color", color);
  probabilityEl.textContent = `${(clamped * 100).toFixed(1)}%`;
  classificationDecisionEl.classList.remove("is-same", "is-different");

  if (showClassification) {
    classificationConfidenceEl.textContent = classificationConfidence(clamped);
    classificationDecisionEl.textContent = decision;
    classificationDecisionEl.classList.add(decisionClass);
  } else {
    classificationConfidenceEl.textContent = "";
    classificationDecisionEl.textContent = "";
  }
}

async function loadMetrics() {
  try {
    const response = await fetch("/metrics");
    const metrics = await response.json();
    document.getElementById("metric-f1").textContent = formatNumber(metrics.f1);
    document.getElementById("metric-youden").textContent = formatNumber(metrics.youden_j);
    document.getElementById("metric-auc").textContent = formatNumber(metrics.auc_roc);
    document.getElementById("metric-f1-rating").textContent = metricRating(metrics.f1);
    document.getElementById("metric-youden-rating").textContent = metricRating(metrics.youden_j);
    document.getElementById("metric-auc-rating").textContent = metricRating(metrics.auc_roc);
  } catch (error) {
    console.error("Failed to load metrics", error);
  }}


async function handlePredict() {
  clearError();

  const text1 = document.getElementById("text1").value.trim();
  const text2 = document.getElementById("text2").value.trim();

  if (!text1 || !text2) {
    showError("Please fill in both text fields.");
    return;
  }

  buttonEl.disabled = true; // disable click
  buttonEl.textContent = "Running...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text1, text2 }),
    });
    const result = await response.json();
    if (!response.ok || result.error) {
      throw new Error(result.error || "Request failed.");
    }
    renderProbability(result.probability);
  } catch (error) {
    showError(error.message || "Request failed.");
  } finally {
    buttonEl.disabled = false;
    buttonEl.textContent = "Predict";
  }}

buttonEl.addEventListener("click", handlePredict);

renderProbability(0, false);
loadMetrics(); // always show performance metrics
