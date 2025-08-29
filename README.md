# 📻 Leveraging Local Radio for Real-Time Food Security Insights: An AI-Powered Approach - Preliminary Phase 🇸🇴  

This project, in collaboration with **Zero Hunger Lab**, analyzes Somali radio broadcasts to extract food security insights using **Speech-to-Text (STT)** and **LLMs**. The goal is to assess whether radio stations provide timely and useful information to improve food security indicators.  

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/AdamPrzychodni/somali-radios-with-ai-for-food-security.git
cd somali-radios-with-ai-for-food-security
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
You can now run the Jupyter notebooks:
```bash
jupyter notebook
```

Navigate to either `1_phase/` or `2_phase/` directories to access the project notebooks.

---

## 📂 Project Structure

```
somali-radios-with-ai-for-food-security/
├── 1_phase/
│   └── Radio_Ergo_Somali_Speech_to_Text_for_Food_Security.ipynb
├── 2_phase/
│   ├── 1_EDA_IPC_Soamli.ipynb
│   ├── 2_data_collection.ipynb
│   ├── 3_transcription.ipynb
│   ├── 4_topic_modeling.ipynb
│   └── 4_update_IPC_based_on_radioergo.ipynb
├── README.md
└── requirements.txt
```

---

## 📂 Project Documents  

### 📑 **Preliminary Research Phase**  
*Leveraging Local Radio for Real-Time Food Security Insights: An AI-Powered Approach - Preliminary Phase.pdf*  

This document outlines the initial phase of the research project, exploring how AI can extract food security insights from Somali radio broadcasts. Key objectives include:  
- Identifying relevant radio stations, with a focus on **Radio Ergo 📻**  
- Collecting audio samples  
- Testing **speech-to-text (STT)** technologies for the Somali language  
- Evaluating the feasibility of enhancing food security monitoring in Somalia, despite **challenges with transcription accuracy**  

---

### 🛠 **Implementation: Speech-to-Text Pipeline**  
*Radio Ergo Somali Speech-to-Text for Food Security.ipynb*  

This Jupyter Notebook implements a pipeline to:  
✅ Download broadcasts from **Radio Ergo's SoundCloud channel**  
✅ Transcribe them using **Whisper** and a **fine-tuned Whisper model for Somali language**

---

## 🔧 Troubleshooting

If you encounter issues during installation:

- **Permission errors**: Try using `pip install --user -r requirements.txt`
- **Package conflicts**: Make sure you're using a virtual environment
- **Python version**: Ensure you have Python 3.8 or higher with `python --version`

---

🌍 This project aims to bridge the gap between **real-time local information** and **food security monitoring**, leveraging AI-driven insights from radio broadcasts.