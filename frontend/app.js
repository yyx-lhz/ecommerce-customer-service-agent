const samples = [
  "Hello, we are looking for cotton canvas tote bags. Please quote USD 2.00 for 1,000 pcs shipping to Australia, FOB. Could you also confirm the production lead time?",
  "Need 500 pcs stainless steel bottles shipping to Canada. Please send your best quotation.",
  "Could you send me your catalog for ceramic mugs?",
];

const form = document.querySelector("#inquiryForm");
const inquiryText = document.querySelector("#inquiryText");
const processButton = document.querySelector("#processButton");
const emptyState = document.querySelector("#emptyState");
const loadingState = document.querySelector("#loadingState");
const resultContent = document.querySelector("#resultContent");
const errorState = document.querySelector("#errorState");
const statusPill = document.querySelector("#statusPill");
let selectedSource = "email";
let sampleIndex = 0;

document.querySelectorAll(".source-tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".source-tab").forEach((item) => item.classList.remove("selected"));
    button.classList.add("selected");
    selectedSource = button.dataset.source;
  });
});

document.querySelector("#loadSample").addEventListener("click", () => {
  inquiryText.value = samples[sampleIndex % samples.length];
  sampleIndex += 1;
  updateCharCount();
  inquiryText.focus();
});

inquiryText.addEventListener("input", updateCharCount);
document.querySelector("#copyReply").addEventListener("click", async (event) => {
  await navigator.clipboard.writeText(document.querySelector("#replyDraft").value);
  event.currentTarget.textContent = "Copied";
  setTimeout(() => { event.currentTarget.textContent = "Copy reply"; }, 1400);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoading(true);
  try {
    const record = await request("/inquiries", {
      method: "POST",
      body: JSON.stringify({
        content: inquiryText.value.trim(),
        source: selectedSource,
        customer_name: valueOrNull("#customerName"),
        customer_email: valueOrNull("#customerEmail"),
      }),
    });
    const result = await request(`/inquiries/${record.inquiry_id}/process`, { method: "POST" });
    renderResult(result);
  } catch (error) {
    showError(error.message || "Unable to reach the inquiry service.");
  } finally {
    setLoading(false);
  }
});

async function request(url, options) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed with status ${response.status}`);
  }
  return response.json();
}

function renderResult(result) {
  const structured = result.inquiry.structured || {};
  const delivery = structured.delivery || {};
  const specs = structured.specifications || {};
  document.querySelector("#inquiryId").textContent = result.inquiry.inquiry_id;
  document.querySelector("#requirementGrid").innerHTML = [
    ["Product", structured.product_type],
    ["Quantity", structured.quantity ? `${formatNumber(structured.quantity)} ${structured.quantity_unit || "pcs"}` : null],
    ["Market", structured.country_or_region],
    ["Target price", structured.target_price ? `${structured.price_currency || ""} ${structured.target_price}` : null],
    ["Incoterm", delivery.incoterm],
    ["Intent", humanize(structured.intent)],
    ["Specifications", specs.free_text && Object.keys(specs.attributes || {}).length ? JSON.stringify(specs.attributes) : "Captured in source message"],
    ["Required date", delivery.required_date],
    ["Shipping", delivery.shipping_method],
  ].map(([label, value]) => `<div class="requirement"><small>${escapeHtml(label)}</small><strong>${escapeHtml(value || "Not provided")}</strong></div>`).join("");

  const completeness = result.completeness;
  const badge = document.querySelector("#completenessBadge");
  const body = document.querySelector("#clarificationBody");
  if (completeness.is_complete) {
    badge.textContent = "Complete";
    body.innerHTML = '<div class="complete-box">✓ Minimum information is available for product matching and quotation.</div>';
  } else {
    badge.textContent = `${completeness.missing_fields.length} fields missing`;
    body.innerHTML = `<div class="missing-box"><strong>Buyer clarification required</strong><ul>${completeness.clarification_questions.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>`;
  }

  const matchSection = document.querySelector("#matchSection");
  const toolsSection = document.querySelector("#toolsSection");
  if (result.match) {
    matchSection.classList.remove("hidden");
    document.querySelector("#matchCard").innerHTML = `<div class="match-top"><div><small>${escapeHtml(result.match.sku)}</small><h4>${escapeHtml(result.match.name)}</h4></div><span class="score">${Math.round(result.match.score * 100)}%</span></div><ul class="reasons">${result.match.reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}</ul>`;
    document.querySelector("#candidateList").innerHTML = result.candidates.map((candidate) => `<div class="candidate"><strong>${escapeHtml(candidate.sku)}</strong><small>${Math.round(candidate.score * 100)}% semantic score</small></div>`).join("");
  } else {
    matchSection.classList.add("hidden");
  }

  if (Object.keys(result.tool_results || {}).length) {
    toolsSection.classList.remove("hidden");
    const tools = result.tool_results;
    const toolItems = [
      ["$", "Price", tools.price ? `${tools.price.currency} ${Number(tools.price.unit_price).toFixed(2)} / pc` : "—"],
      ["▦", "Inventory", tools.inventory ? `${formatNumber(tools.inventory.available_quantity)} pcs` : "—"],
      ["◷", "Lead time", tools.lead_time ? `${tools.lead_time.production_lead_time_days} days` : "—"],
      ["⌁", "MOQ", tools.inventory ? `${formatNumber(tools.inventory.moq)} pcs` : "—"],
    ];
    document.querySelector("#toolGrid").innerHTML = toolItems.map(([icon, label, value]) => `<div class="tool-card"><div class="tool-icon">${icon}</div><small>${label}</small><strong>${escapeHtml(value)}</strong></div>`).join("");
  } else {
    toolsSection.classList.add("hidden");
  }

  document.querySelector("#replyDraft").value = result.reply_draft;
  document.querySelector("#traceList").innerHTML = result.trace.map((step, index) => `<div class="trace-node"><span>${escapeHtml(humanize(step))}</span>${index < result.trace.length - 1 ? "<b>→</b>" : ""}</div>`).join("");

  emptyState.classList.add("hidden");
  errorState.classList.add("hidden");
  resultContent.classList.remove("hidden");
  statusPill.textContent = humanize(result.status);
  statusPill.className = `status-pill ${result.status === "ready_for_reply" ? "ready" : "waiting"}`;
}

function setLoading(isLoading) {
  processButton.disabled = isLoading;
  processButton.querySelector("span").textContent = isLoading ? "Agent is working…" : "Process inquiry";
  loadingState.classList.toggle("hidden", !isLoading);
  if (isLoading) {
    emptyState.classList.add("hidden");
    resultContent.classList.add("hidden");
    errorState.classList.add("hidden");
    statusPill.textContent = "Processing";
    statusPill.className = "status-pill processing";
  }
}

function showError(message) {
  emptyState.classList.add("hidden");
  resultContent.classList.add("hidden");
  errorState.classList.remove("hidden");
  document.querySelector("#errorMessage").textContent = message;
  statusPill.textContent = "Failed";
  statusPill.className = "status-pill waiting";
}

function valueOrNull(selector) {
  const value = document.querySelector(selector).value.trim();
  return value || null;
}

function updateCharCount() {
  document.querySelector("#charCount").textContent = `${inquiryText.value.length.toLocaleString()} / 20,000`;
}

function humanize(value) {
  if (!value) return "Not classified";
  return String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatNumber(value) { return Number(value).toLocaleString("en-US"); }

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}
