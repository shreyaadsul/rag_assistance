# 🚀 Advanced RAG Observability Dashboard with Research-Grade Telemetry & Multi-Metric Evaluation

An advanced, premium research-oriented **Retrieval-Augmented Generation (RAG) Observability Platform** built using **Flask**, **FAISS**, **HuggingFace Embeddings**, and the high-speed **Groq LLM Engine**.

This project elevates traditional RAG from a standard conversational bot interface into a state-of-the-art enterprise observability space. Designed with a professional **2-column SaaS application layout**, it empowers researchers to ingest documents, track sub-second pipeline processing speeds, evaluate vector distance retrieval parameters, interact with collapsible context blocks, and cross-reference evaluation performance using pure-CSS visual telemetry graphs.

---

## 🧠 Core Architecture & Enterprise Capabilities

### 1. 📐 Premium 2-Column Workspace Topology
- **Left Panel (Knowledge Base & Telemetry)**: Dedicated strictly to multi-part document ingestion, mounted PDF diagnostics, and real-time benchmark timing loops. 
- **Right Panel (AI Research Workspace)**: Hosts query construction logic coupled with a dedicated fixed-height multi-tab observability pane, ensuring complete elimination of vertical clutter and excessive document scrolling.

### 2. 🗂️ Advanced Multi-Tab Observability System
Users can instantly toggle specific diagnostic views within the output bounding frame:
- **[ Overview ]**: Features the primary grounded synthesis response along with a compact status ribbon displaying the overarching reliability verdict and contextual grounding confidence meter.
- **[ Retrieval ]**: Hosts interactive chunk analysis, dynamic match-quality tier indicators, and relevance progress distributions.
- **[ Evaluation ]**: Formatted as an aligned **4-column horizontal mini-cards grid** capturing discrete correctness scales, validation bounds, and completeness profiles.
- **[ Analysis ]**: Contains both the raw text reasoning logs and the integrated dynamic telemetry graph.

### 3. 🪗 Collapsible Retrieval Accordions with Restricted Buffers
- **Sleek Row Ingestion**: Retrieved segments are structured as custom mini-cards featuring relevance meters and source page indicators.
- **Internal Scroll Enclosure**: Default text previews are automatically clipped at exactly `120 characters` (`preview[:120] + "..."`). Selecting a chunk header smoothly toggles an expanded internal buffer block with independent `max-height` constraints, making deep segment verification highly intuitive.
- **Match-Tier Mapping**:
  - **90–100%**: Excellent Match (🟢 Emerald Green Gradient)
  - **70–89%**: Strong Match (🔵 Sky Blue Gradient)
  - **50–69%**: Moderate Match (🟡 Amber Yellow Gradient)
  - **Below 50%**: Weak Match (🔴 Rose Red Gradient)

### 4. 🔬 Research-Grade Answer Evaluation Framework
Enforces strict JSON schema validation comparing generated text directly against retrieved context to assign:
- **Correctness Score** ($0\text{--}10$)
- **Contextual Grounding Confidence** ($0\text{--}100\%$)
- **Hallucination Risk Tracker** (`Low` / `Medium` / `High`)
- **Response Completeness Tier** (`Low` / `Medium` / `High`)
- **Overall Verdict** (`Reliable` / `Partially Reliable` / `Unreliable`)
- **Detailed Analytical Justification**

### 5. 📊 RAG Lifecycle Observability Graph
Embedded directly inside the **Analysis** tab, a custom pure-CSS multi-column telemetry chart maps real-time cross-stage metrics side-by-side:
- **Avg Retrieval Relevance**: The combined mean similarity score of candidates returned.
- **Grounding Confidence**: Evaluation module confidence scoring track.
- **Synthesis Correctness**: Converted absolute correctness parameter ($10 \times \text{Score}$).
- **Global Pipeline Index**: A weighted aggregate index summarizing generation integrity ($35\%$ retrieval, $35\%$ confidence, $30\%$ correctness).
- Accompanied by absolute dashed horizontal guides tracking $0\%$ to $100\%$ validation intervals.

### 6. ⚡ Sub-Second Benchmarking Telemetry
- Active pipeline execution features smooth CSS looping typing animations (`Retrieving context...`).
- Integrates live continuous decimal execution tracking intervals detailing exactly how long each phase (Ingestion, Generation, Validation) operates down to the millisecond (`0.42s`, `1.83s`).

---

## ⚙️ Execution Pipeline Topology

```text
       [ User Query Submission ]
                   │
                   ▼
       [ HuggingFace Embeddings ]
                   │
                   ▼
     [ FAISS Vector Distance Search ] ──▶ ( Top K=5 Distance-Scored Candidate Chunks )
                   │                                         │
                   ▼                                         ▼
         [ Groq Synthesizer ]                     [ Retrieval Intelligence ]
                   │                              ( Collapsible Accordions )
                   ▼                                         │
       [ Validation Evaluator ]                              │
   ( Correctness, Confidence, Graph )                        │
                   │                                         │
                   ▼                                         ▼
   ═══════════════════════════════════════════════════════════════════════
   [ Complete Unified Presentation DOM Handover via Multi-Tab Workspace  ]
```

---

## 🛠️ Technological Foundation

- **Backend Logic**: Python 3, Flask Web Server
- **AI Vector Frameworks**: LangChain, FAISS Vector Database
- **Embedding Subsystem**: HuggingFace Local Space Embeddings (`all-MiniLM-L6-v2`)
- **Large Language Engine**: Groq Accelerated LLM API
- **Frontend Orchestration**: HTML5 Semantic Layouts, Vanilla CSS Glassmorphism Variables System, ES6 Asynchronous Event Handling

---

## 📂 Repository Topology

```text
rag_assistant/
│
├── app.py                  # Core backend routing, distance normalization, JSON API delivery
├── utils.py                # LangChain FAISS initialization, prompt orchestration mappers
├── requirements.txt        # Verified production dependencies mapping
├── templates/
│   └── index.html          # Professional 2-column client workspace layout templates
├── static/
│   └── style.css           # Premium sci-fi UI variables, bar charting tracks, grid tokens
├── uploads/                # Ephemeral active document staging persistence
└── data/                   # Local serialized FAISS vector space storage
```

---

## ▶️ Setup & Local Deployment

### 1. Clone the Active Project
```bash
git clone <your-repo-link>
cd rag_assistant
```

### 2. Install Required Modules
```bash
pip install -r requirements.txt
```

### 3. Establish Authentication Parameters
Create a localized `.env` configuration file inside the workspace root:
```env
GROQ_API_KEY=your_production_groq_key_here
```

### 4. Invoke Dev Engine locally
```bash
python app.py
```
Access the premium research dashboard interface locally at:
```text
http://127.0.0.1:5000
```

---

## 👩‍💻 Author
**Shreya Adsul**  
TY B.Sc. CS (AIML)

