# 📻 Leveraging Local Radio for Real-Time Food Security Insights: An AI-Powered Approach - Preliminary Phase 🇸🇴  

This project, in collaboration with **Zero Hunger Lab**, analyzes Somali radio broadcasts to extract food security insights using **Speech-to-Text (STT)** and **LLMs**. The goal is to assess whether radio stations provide timely and useful information to improve food security indicators.  

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git
- Internet connection for API access

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

# Additional packages for enhanced transcription
pip install google-generativeai yt-dlp pydub
```

### 4. Setup Gemini API Key 🔑

The project now supports **Google Gemini API** for enhanced transcription accuracy. Follow these steps:

#### 4.1 Create Your API Key
1. **Visit Google AI Studio**: Go to [https://aistudio.google.com](https://aistudio.google.com)
2. **Sign in** with your Google Account
3. **Accept terms**: Accept the Google APIs Terms of Service and Gemini API Additional Terms
4. **Create API key**: Click "Get API key" → "Create API key" 
5. **Choose project**: Select existing Google Cloud project or create new one
6. **Copy key**: Save the generated API key securely

#### 4.2 Configure Environment Variable
Choose one of these methods:

**Method A: Environment Variable (Recommended)**
```bash
# Check your shell
echo $SHELL

cd somali-radios-with-ai-for-food-security

# If using bash
nano ~/.bashrc
# Add this line:
export GEMINI_API_KEY="your_api_key_here"
# Save and apply:
source ~/.bashrc

# If using zsh  
nano ~/.zshrc
# Add this line:
export GEMINI_API_KEY="your_api_key_here"
# Save and apply:
source ~/.zshrc
```

**Method B: .env File (Project-specific)**
```bash
cd somali-radios-with-ai-for-food-security

# Create .env file in project root
echo 'GEMINI_API_KEY="your_api_key_here"' > .env
```

#### 4.3 Install Audio Processing Tools
```bash
# Install ffmpeg (required for audio processing)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# Fedora:
sudo dnf install ffmpeg

# Arch:
sudo pacman -S ffmpeg
```

#### 4.4 Verify Setup
Run the verification script to ensure everything works:
```python
# Test your setup
from setup_verification import run_full_setup_test
run_full_setup_test()
```

### 5. Verify Installation
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
├── enhanced_downloader.py       # 🆕 New streaming transcription module
├── setup_verification.py        # 🆕 API setup verification
├── .env.example                # 🆕 Environment template
├── README.md
└── requirements.txt
```

---

## 🆕 New Features: On-the-Fly Transcription

### Stream Processing Workflow
The enhanced system now supports **memory-efficient transcription** without storing audio files:

```
Download Audio → Load to Memory → Transcribe → Save Text → Cleanup
```

### Key Capabilities
- **🎯 Stream Processing**: No temporary file storage required
- **🧠 Multiple AI Engines**: Support for both Gemini API and Whisper
- **💾 Memory Efficient**: Automatic cleanup and batch processing
- **📝 Smart Logging**: Tracks processing history and prevents duplicates

### Quick Usage Examples

**Single URL Transcription:**
```python
from enhanced_downloader import stream_transcribe_single_url

success, transcript_path = stream_transcribe_single_url(
    url="https://soundcloud.com/radio-ergo/idaacadda-01-jul-2024",
    output_dir="transcripts"
)
```

**Date Range Processing:**
```python
from enhanced_downloader import stream_transcribe_date_range

results = stream_transcribe_date_range(
    profile_url="https://soundcloud.com/radio-ergo",
    start_date="2024-07-01",
    end_date="2024-07-31",
    output_dir="july_transcripts"
)
```

**Existing MP3 Files:**
```python
from enhanced_downloader import batch_transcribe_existing_mp3s

results = batch_transcribe_existing_mp3s(
    input_dir="./audio_files",
    output_dir="transcripts",
    batch_size=3
)
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

### 🛠 **Implementation: Speech-to-Text Pipeline**  
*Radio Ergo Somali Speech-to-Text for Food Security.ipynb*  

This Jupyter Notebook implements a pipeline to:  
✅ Download broadcasts from **Radio Ergo's SoundCloud channel**  
✅ Transcribe them using **Whisper** and a **fine-tuned Whisper model for Somali language**
✅ **NEW**: Enhanced with Gemini API for improved transcription accuracy
✅ **NEW**: Memory-efficient streaming processing

---

## 🔧 Troubleshooting

### Installation Issues
If you encounter issues during installation:
- **Permission errors**: Try using `pip install --user -r requirements.txt`
- **Package conflicts**: Make sure you're using a virtual environment
- **Python version**: Ensure you have Python 3.8 or higher with `python --version`

### API Setup Issues
If Gemini API is not working:
- **API Key**: Verify your key is correct and active at [Google AI Studio](https://aistudio.google.com)
- **Internet**: Ensure stable internet connection
- **Quota**: Check you haven't exceeded free tier limits
- **Environment**: Run the verification script to diagnose issues

### Audio Processing Issues
If audio downloads or transcription fail:
- **ffmpeg**: Ensure ffmpeg is installed and accessible
- **Permissions**: Check file system permissions for output directories
- **Memory**: For large files, increase batch size or reduce concurrent processing

### Common Error Solutions
```bash
# Fix missing ffmpeg
sudo apt-get update && sudo apt-get install ffmpeg

# Fix Python package issues
pip install --upgrade pip
pip install --force-reinstall google-generativeai

# Reset environment
deactivate && source venv/bin/activate
```

---

## 🌍 Impact & Goals

This project aims to bridge the gap between **real-time local information** and **food security monitoring**, leveraging AI-driven insights from radio broadcasts to:

- **Enhance early warning systems** for food security crises
- **Provide timely information** to humanitarian organizations
- **Support evidence-based decision making** for food assistance programs
- **Demonstrate AI applications** in humanitarian contexts

### Research Questions
- Can AI accurately transcribe Somali radio broadcasts?
- Do radio stations provide actionable food security information?
- How can local media complement traditional monitoring systems?

---

## 📊 Data Privacy & Ethics

- All processing respects copyright and fair use guidelines
- No personal data is collected from radio broadcasts
- API usage follows Google's terms of service
- Research conducted under academic ethics protocols

---

## 🤝 Contributing

This is an academic research project. For questions or collaboration:
- **Contact**: Adam Przychodni
- **Institution**: Zero Hunger Lab collaboration
- **Purpose**: Food security research and humanitarian applications

---

## 📄 License

This project is for academic research purposes. Please respect copyright of original radio content and API terms of service.

---

**🎯 Next Steps**: Run the setup verification, configure your API key, and start exploring the enhanced transcription capabilities!