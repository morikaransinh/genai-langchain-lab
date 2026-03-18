# 🧠 Healthcare Career Intelligence System

An AI-powered platform that analyzes resumes, identifies skill gaps, and provides personalized learning recommendations for healthcare technology careers.

---

## 🚀 Live Demo

* 🔗 **Backend API**: https://genai-langchain-lab.onrender.com
* 📄 **Swagger Docs**: https://genai-langchain-lab.onrender.com/docs

---

## 💡 Key Features

* 📄 **Resume Parsing** – Upload PDF resumes for automated analysis
* 🧠 **AI Skill Extraction** – Extract relevant skills using LLMs
* 📊 **Skill Gap Analysis** – Compare current skills with target roles
* 🎯 **Career Goal Matching** – Align resume with healthcare tech roles
* 📚 **Learning Recommendations** – Get curated resources to bridge gaps
* ⚡ **FastAPI Backend** – High-performance API system
* 🎨 **Streamlit Frontend** – Simple and interactive UI

---

## 🏗️ Tech Stack

### 🔧 Backend

* **FastAPI** – API development
* **LangChain** – LLM orchestration
* **Groq API** – High-speed LLM inference
* **PyMuPDF** – PDF parsing
* **SerpAPI** – Resource recommendation

### 🎨 Frontend

* **Streamlit** – Rapid UI development

### ☁️ Deployment

* **Render** – Backend hosting

---

## 📂 Project Structure

```
.
├── skill_gap_analyser/
│   ├── app.py                  # Streamlit frontend
│   ├── main.py                 # FastAPI entry point
│   ├── models.py               # Pydantic models
│   ├── services.py             # Core logic
│   ├── career_gap_models.py    # Career-specific schemas
│   └── career_gap_service.py   # Gap analysis logic
│
├── .env
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Configure Environment Variables

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
SERPAPI_KEY=your_serpapi_key
```

---

## ▶️ Run Locally

### Backend

```
uvicorn skill_gap_analyser.main:app --reload
```

### Frontend

```
streamlit run skill_gap_analyser/app.py
```

---

## 🧪 Usage

1. Upload a **PDF resume**
2. Enter your **target role** (e.g., *Healthcare Data Engineer*)
3. Click **Analyze**

### 🔍 Output Includes:

* Extracted Skills
* Missing Skills (Gap Analysis)
* Gap Percentage
* Learning Resources
* Personalized Recommendations

---

## 📌 API Endpoint

### 🔹 POST `/predict`

Extracts structured insights from a resume and returns:

* Skills
* Career match
* Skill gaps
* Recommendations

---

## 🔮 Future Improvements

* 🔄 Multi-resume support with memory (RAG pipelines)
* 📈 Skill progress tracking dashboard
* 🗺️ Career roadmap visualization (like roadmap.sh)
* 👤 User authentication & saved reports
* 🌐 Full frontend deployment

---

## 🧠 Why This Project?

This project demonstrates:

* Real-world application of **LLMs in career intelligence**
* End-to-end system design (**API + UI + Deployment**)
* Practical use of **RAG + AI + Resume Parsing**
* Focus on **healthcare tech domain**, a rapidly growing industry

---

## 🤝 Contributing

Feel free to fork the repo and submit pull requests!

---

## 📬 Contact

If you’d like to collaborate or discuss opportunities, feel free to connect 🚀
