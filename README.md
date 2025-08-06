# AI Podcast Clipper 🎹🎨

Automatically convert full podcast episodes into viral, short-form clips optimized for platforms like **YouTube Shorts** and **TikTok**. This AI-powered SaaS application intelligently transcribes podcasts, detects viral moments, identifies the active speaker, and renders vertical clips ready to publish.

---

## 🔗 Live Demo

🔗 [Visit the Live App](https://frontend-five-snowy-51.vercel.app)

---

## 🧠 Overview

This is a full-stack, production-ready SaaS platform that empowers podcast creators to generate engaging, viral-ready clips from long-form videos with minimal effort. Leveraging a suite of AI models and serverless GPU processing, this tool handles everything from transcription to video cropping, rendering, and delivering the final clip to users.

---

## 🚀 Features

* 🎥 **Auto-detects viral moments** (stories, questions, high engagement points)
* 🔊 **Subtitles** added automatically to clips
* 📄 **Transcription** via `m-bain/whisperX`
* 👩‍🎤 **Speaker face detection** using [LR-ASD](https://github.com/Junhua-Liao/LR-ASD)
* 📱 **Vertical video format** for TikTok & Shorts
* ⏹️ **GPU-accelerated rendering** via FFMPEGCV
* 🧠 **LLM-powered moment identification** using Gemini API
* 📊 **Queue system** using Inngest for background processing
* 💳 **Credit-based** clip generation system
* 💸 **Stripe integration** for purchasing credits
* 👤 **User authentication** with Auth.js
* 📊 **Dashboard** to upload, track and preview podcast clips
* ⏱️ Long-running task handling using **Inngest**
* ✨ Serverless GPU-powered inference using **Modal**
* 🛠️ **FastAPI** endpoint to manage podcast processing pipeline
* 🌟 **Beautiful UI** with Tailwind CSS + ShadCN

---

## ⚙️ Tech Stack

### 🖥️ Frontend

* Next.js 15
* React + TypeScript
* Tailwind CSS
* ShadCN UI
* Framer Motion
* Auth.js

### 🧬 Backend

* FastAPI (Python)
* FFMPEGCV for rendering
* Modal for GPU tasks
* Inngest for background jobs
* WhisperX for transcription
* Gemini API (Google AI)
* AWS S3 for video storage

### 💳 Payments & Database

* Stripe for credit-based billing
* Prisma + PostgreSQL for database management

---

## 💪 Running Locally

### ⚡ Requirements

* Python 3.12.3
* Node.js 18+
* PostgreSQL Database
* Stripe credentials
* Modal, Inngest API keys

### 📂 Clone Project

```bash
git clone https://github.com/rashap224/ai-podcast-clips.git
cd ai-podcast-clips
```

### 🛠️ Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
python main.py
```

### 🗱 Frontend Setup

```bash
cd frontend
npm install
npx prisma generate
npm run dev
```

### 🔗 Local URLs

* Frontend: [http://localhost:3000](http://localhost:3000)
* Backend: [http://localhost:8000](http://localhost:8000)

---

## 🧬 LR-ASD Model

This project uses [LR-ASD](https://github.com/Junhua-Liao/LR-ASD) for active speaker detection. It allows accurate cropping and tracking of the speaker's face to keep the visual focus engaging.

---

## 📊 Project Structure

```
ai-podcast-clips/
├── backend/
│   ├── LR-ASD/              # Speaker Detection Model
│   ├── main.py              # FastAPI Backend
│   └── ...
├── frontend/               # Next.js + Tailwind
│   ├── app/
│   ├── components/
│   └── ...
├── .gitignore
├── README.md
└── ...
```

---

## 📈 Future Roadmap

* Add multilingual subtitle support
* Experiment with viral moment identification prompts to extract interesting parts from videos, I ll try using different Gemini models
* Add multiple modes (e.g., “question mode”, “story mode”) via a frontend dropdown, processed differently in the backend.
* Enable YouTube link input to auto-download videos using tools like pytubefix or yt-dlp, avoiding manual uploads.
* Allow YouTube channel connection to automatically fetch videos from the user's linked channel.
* Use a CRON job to auto-schedule, clip, render, and post one video clip to the user's YouTube channel daily.
* Render low-resolution clip previews for faster browsing, with an option to re-render selected clips in 1920×1080.

---

## 🙌 Credits

* [LR-ASD](https://github.com/Junhua-Liao/LR-ASD)
* [WhisperX](https://github.com/m-bain/whisperX)
* [Modal](https://modal.com)
* [Inngest](https://www.inngest.com/)
* [Auth.js](https://authjs.dev/)
* [Stripe](https://stripe.com)
* [Tailwind CSS](https://tailwindcss.com)
* [Prisma](https://www.prisma.io)
* [PostgreSQL](https://www.postgresql.org)

---

