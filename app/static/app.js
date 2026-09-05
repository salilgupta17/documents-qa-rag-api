document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("file-input");
  const fileLabel = document.getElementById("file-label");
  const uploadForm = document.getElementById("upload-form");
  const uploadBtn = document.getElementById("upload-btn");
  const uploadError = document.getElementById("upload-error");
  const uploadSuccess = document.getElementById("upload-success");
  
  const documentsList = document.getElementById("documents-list");
  const docCountBadge = document.getElementById("doc-count-badge");
  const docFilter = document.getElementById("doc-filter");
  
  const queryForm = document.getElementById("query-form");
  const questionInput = document.getElementById("question-input");
  const askBtn = document.getElementById("ask-btn");
  const queryError = document.getElementById("query-error");
  const chatFeed = document.getElementById("chat-feed");
  
  const statusText = document.getElementById("status-text");

  // File selection UI feedback
  fileInput.addEventListener("change", (e) => {
    if (fileInput.files.length > 0) {
      fileLabel.textContent = fileInput.files[0].name;
    } else {
      fileLabel.textContent = "Choose a PDF or TXT file";
    }
  });

  // Health check & System Status
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        statusText.textContent = `Online (${data.mode.toUpperCase()} Mode) • ${data.total_chunks} Chunks`;
      } else {
        statusText.textContent = "Server Degraded";
      }
    } catch (err) {
      statusText.textContent = "Offline / Error";
    }
  }

  // Load Ingested Documents
  async function loadDocuments() {
    try {
      const res = await fetch("/documents");
      if (!res.ok) throw new Error("Failed to load documents list");
      const data = await res.json();
      const docs = data.documents || [];
      
      docCountBadge.textContent = `${docs.length} Doc${docs.length === 1 ? '' : 's'}`;
      
      // Populate left list
      if (docs.length === 0) {
        documentsList.innerHTML = '<p class="empty-state">No documents uploaded yet.</p>';
      } else {
        documentsList.innerHTML = docs.map(doc => `
          <div class="doc-item" data-id="${doc.document_id}">
            <div class="doc-info">
              <span class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
              <span class="doc-meta">${doc.chunk_count} chunks</span>
            </div>
            <button type="button" class="btn-delete" data-id="${doc.document_id}">Delete</button>
          </div>
        `).join('');

        // Add delete event listeners
        document.querySelectorAll(".btn-delete").forEach(btn => {
          btn.addEventListener("click", (e) => {
            const id = e.target.getAttribute("data-id");
            deleteDocument(id);
          });
        });
      }

      // Populate filter dropdown
      const currentSelected = docFilter.value;
      docFilter.innerHTML = '<option value="">All Documents</option>' + 
        docs.map(doc => `<option value="${doc.document_id}">${escapeHtml(doc.filename)}</option>`).join('');
      docFilter.value = currentSelected;

    } catch (err) {
      console.error("Error loading documents:", err);
    }
  }

  // Handle Document Upload
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideBanners();
    
    if (!fileInput.files || fileInput.files.length === 0) {
      showError(uploadError, "Please select a file to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    setLoading(uploadBtn, true, "Uploading...");

    try {
      const res = await fetch("/documents/upload", {
        method: "POST",
        body: formData
      });

      const data = await res.json();

      if (!res.ok) {
        showError(uploadError, data.detail || "Upload failed.");
      } else {
        showSuccess(uploadSuccess, `Ingested '${data.filename}' (${data.chunk_count} chunks).`);
        fileInput.value = "";
        fileLabel.textContent = "Choose a PDF or TXT file";
        await loadDocuments();
        await checkHealth();
      }
    } catch (err) {
      showError(uploadError, "Network error during upload: " + err.message);
    } finally {
      setLoading(uploadBtn, false, "Upload & Index");
    }
  });

  // Handle Document Delete
  async function deleteDocument(docId) {
    if (!confirm("Are you sure you want to delete this document and remove its vectors from the FAISS index?")) {
      return;
    }

    try {
      const res = await fetch(`/documents/${docId}`, {
        method: "DELETE"
      });

      const data = await res.json();
      if (!res.ok) {
        showError(uploadError, data.detail || "Delete failed.");
      } else {
        showSuccess(uploadSuccess, data.message || "Document deleted.");
        await loadDocuments();
        await checkHealth();
      }
    } catch (err) {
      showError(uploadError, "Error deleting document: " + err.message);
    }
  }

  // Handle Question Query
  queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideBanners();

    const question = questionInput.value.trim();
    if (!question) return;

    const docIdFilter = docFilter.value || null;

    // Clear welcome message if present
    const welcome = document.querySelector(".welcome-message");
    if (welcome) welcome.remove();

    // Create chat card with loading state
    const cardId = "card-" + Date.now();
    const card = document.createElement("div");
    card.className = "chat-card";
    card.id = cardId;
    card.innerHTML = `
      <div class="question-text">Q: ${escapeHtml(question)}</div>
      <div class="answer-text" id="ans-${cardId}"><em>Thinking…</em></div>
      <div class="sources-container hidden" id="src-${cardId}">
        <div class="sources-title">Sources</div>
        <div class="sources-list" id="srclist-${cardId}"></div>
      </div>
    `;
    chatFeed.appendChild(card);
    chatFeed.scrollTop = chatFeed.scrollHeight;

    questionInput.value = "";
    setLoading(askBtn, true, "Asking...");

    try {
      const res = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question,
          document_id: docIdFilter
        })
      });

      const data = await res.json();

      const ansElem = document.getElementById(`ans-${cardId}`);
      const srcContainer = document.getElementById(`src-${cardId}`);
      const srcList = document.getElementById(`srclist-${cardId}`);

      if (!res.ok) {
        ansElem.innerHTML = `<span style="color: var(--danger)">Error: ${escapeHtml(data.detail || "Query failed.")}</span>`;
      } else {
        ansElem.textContent = data.answer;

        if (data.sources && data.sources.length > 0) {
          srcContainer.classList.remove("hidden");
          srcList.innerHTML = data.sources.map(s => `
            <div class="source-item">
              <span class="source-tag">${escapeHtml(s.filename)} (Page ${s.page})</span>: "${escapeHtml(s.snippet)}"
            </div>
          `).join('');
        }
      }
    } catch (err) {
      showError(queryError, "Failed to connect to query endpoint: " + err.message);
    } finally {
      setLoading(askBtn, false, "Ask");
      chatFeed.scrollTop = chatFeed.scrollHeight;
    }
  });

  // Helpers
  function setLoading(btn, isLoading, text) {
    btn.disabled = isLoading;
    btn.querySelector("span").textContent = text;
  }

  function showError(elem, msg) {
    elem.textContent = msg;
    elem.classList.remove("hidden");
  }

  function showSuccess(elem, msg) {
    elem.textContent = msg;
    elem.classList.remove("hidden");
    setTimeout(() => elem.classList.add("hidden"), 4000);
  }

  function hideBanners() {
    uploadError.classList.add("hidden");
    uploadSuccess.classList.add("hidden");
    queryError.classList.add("hidden");
  }

  function escapeHtml(str) {
    return (str || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // Initial loads
  checkHealth();
  loadDocuments();
});
