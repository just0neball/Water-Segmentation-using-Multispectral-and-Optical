
"use strict";

const $ = (id) => document.getElementById(id);

const galleryGrid = $("galleryGrid");
const galleryCount = $("galleryCount");
const gallerySearch = $("gallerySearch");
const selectedText = $("selectedText");
const galleryButton = $("analyzeGalleryBtn");

const fileInput = $("imageFile");
const fileName = $("fileName");
const uploadButton = $("uploadBtn");

const statusText = $("status");
const statusIndicator = $("statusIndicator");
const resultTitle = $("resultTitle");
const coverage = $("coverage");
const resultSource = $("resultSource");
const resultFile = $("resultFile");

let gallerySamples = [];
let selectedSampleId = null;
let busy = false;

function setStatus(message, state = "READY") {
  statusText.textContent = message;
  statusIndicator.textContent = state;
}

function setBusy(value) {
  busy = value;
  galleryButton.disabled = busy || !selectedSampleId;
  uploadButton.disabled = busy || !fileInput.files.length;
}

function resetResults() {
  for (const key of ["original", "mask", "overlay"]) {
    const image = $(key + "Image");
    const placeholder = $(key + "Placeholder");

    image.hidden = true;
    image.removeAttribute("src");
    placeholder.hidden = false;
  }

  for (const key of ["Original", "Mask", "Overlay"]) {
    const link = $("download" + key);
    link.hidden = true;
    link.removeAttribute("href");
  }

  coverage.textContent = "—";
  resultSource.textContent = "—";
  resultFile.textContent = "No file analyzed";
  resultTitle.textContent = "Waiting for analysis";
}

function showResultImage(key, dataUrl, filename) {
  const image = $(key + "Image");
  const placeholder = $(key + "Placeholder");

  image.src = dataUrl;
  image.hidden = false;
  placeholder.hidden = true;

  const title = key.charAt(0).toUpperCase() + key.slice(1);
  const link = $("download" + title);

  link.href = dataUrl;
  link.download = filename;
  link.hidden = false;
}

function displayResult(data) {
  resetResults();

  const baseName = data.sample_id
    ? "sample_" + data.sample_id
    : "uploaded_satellite";

  showResultImage(
    "original",
    data.original_image,
    baseName + "_original.png"
  );

  showResultImage(
    "mask",
    data.mask_image,
    baseName + "_mask.png"
  );

  showResultImage(
    "overlay",
    data.overlay_image,
    baseName + "_overlay.png"
  );

  coverage.textContent = Number(data.water_percentage).toFixed(2) + "%";
  resultSource.textContent = data.source === "gallery"
    ? "Gallery"
    : "Upload";

  resultFile.textContent = data.filename || "Satellite image";

  resultTitle.textContent = data.sample_id
    ? "Sample " + data.sample_id + " analyzed"
    : "Uploaded image analyzed";

  setStatus("Segmentation completed successfully.", "COMPLETED");

  $("resultTitle").scrollIntoView({
    behavior: "smooth",
    block: "center"
  });
}

function createGalleryCard(sample) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "gallery-card";
  button.dataset.sampleId = String(sample.sample_id);

  const image = document.createElement("img");
  image.src = sample.preview_url;
  image.alt = "Satellite sample " + sample.sample_id;
  image.loading = "lazy";

  const title = document.createElement("strong");
  title.textContent = "Sample " + sample.sample_id;

  const description = document.createElement("small");
  description.textContent = "TEST DATASET";

  button.append(image, title, description);

  button.setAttribute(
    "aria-pressed",
    String(selectedSampleId === String(sample.sample_id))
  );

  if (selectedSampleId === String(sample.sample_id)) {
    button.classList.add("selected");
  }

  button.addEventListener("click", () => {
    if (busy) return;

    selectedSampleId = String(sample.sample_id);

    document.querySelectorAll(".gallery-card").forEach(card => {
      const isSelected = card.dataset.sampleId === selectedSampleId;
      card.classList.toggle("selected", isSelected);
      card.setAttribute("aria-pressed", String(isSelected));
    });

    selectedText.textContent = "Selected: Sample " + selectedSampleId;
    galleryButton.disabled = false;

    setStatus(
      "Sample " + selectedSampleId + " selected. Ready for analysis."
    );
  });

  return button;
}

function renderGallery() {
  const query = gallerySearch.value.trim().toLowerCase();

  const filtered = gallerySamples.filter(sample =>
    String(sample.sample_id).toLowerCase().includes(query)
  );

  galleryGrid.replaceChildren();

  if (!filtered.length) {
    const message = document.createElement("p");
    message.className = "gallery-empty";
    message.textContent = "No samples match your search.";
    galleryGrid.appendChild(message);
  } else {
    const fragment = document.createDocumentFragment();

    filtered.forEach(sample => {
      fragment.appendChild(createGalleryCard(sample));
    });

    galleryGrid.appendChild(fragment);
  }

  galleryCount.textContent =
    filtered.length + " / " + gallerySamples.length + " SAMPLES";
}

async function loadGallery() {
  try {
    const response = await fetch("/gallery");

    if (!response.ok) {
      throw new Error("Gallery endpoint returned " + response.status);
    }

    const data = await response.json();

    if (!Array.isArray(data.samples)) {
      throw new Error("Invalid gallery response.");
    }

    gallerySamples = data.samples;
    renderGallery();
  } catch (error) {
    galleryCount.textContent = "UNAVAILABLE";
    galleryGrid.textContent = "Failed to load gallery: " + error.message;
  }
}

async function handleResponse(response) {
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Prediction failed.");
  }

  if (!data.success) {
    throw new Error("The API did not return a successful prediction.");
  }

  return data;
}

galleryButton.addEventListener("click", async () => {
  if (!selectedSampleId || busy) return;

  const sampleId = selectedSampleId;
  setBusy(true);
  resetResults();

  setStatus(
    "Running model inference on Sample " + sampleId + "...",
    "PROCESSING"
  );

  try {
    const response = await fetch(
      "/predict-sample/" + encodeURIComponent(sampleId),
      { method: "POST" }
    );

    const data = await handleResponse(response);
    displayResult(data);
  } catch (error) {
    setStatus("Error: " + error.message, "FAILED");
  } finally {
    setBusy(false);
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];

  fileName.textContent = file ? file.name : "Choose an image file";
  uploadButton.disabled = busy || !file;

  if (file) {
    setStatus("File selected: " + file.name + ". Ready for analysis.");
  }
});

uploadButton.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file || busy) return;

  if (file.size > 20 * 1024 * 1024) {
    setStatus("File exceeds the 20 MB upload limit.", "FAILED");
    return;
  }

  setBusy(true);
  resetResults();
  setStatus("Uploading image and running inference...", "PROCESSING");

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const data = await handleResponse(response);
    displayResult(data);
  } catch (error) {
    setStatus("Error: " + error.message, "FAILED");
  } finally {
    setBusy(false);
  }
});

gallerySearch.addEventListener("input", renderGallery);

$("clearSearch").addEventListener("click", () => {
  gallerySearch.value = "";
  renderGallery();
  gallerySearch.focus();
});

resetResults();
loadGallery();
