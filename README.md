# OpenEnv Customer Support

A simulated customer support environment built with FastAPI. This project provides an API-based system for handling customer support tickets, including classification, refunds, and escalation workflows.

---

## 🚀 Features

* 📦 Simulated customer support environment
* 🧠 Task-based interactions (classification, refund, escalation)
* 🔄 Step-based environment (reset → step → done)
* ⚡ FastAPI-powered REST API
* 🌐 CORS enabled (can connect with frontend easily)
* 🐳 Docker support (if using Dockerfile)

---

## 🏗️ Project Structure

```
.
├── env/
│   ├── environment.py   # Core environment logic
│   ├── models.py        # Data models (Action, etc.)
│   ├── tasks.py         # Task definitions
│   └── graders.py       # Evaluation logic
├── data/
│   └── support_tickets.csv
├── server.py            # FastAPI server
├── inference.py         # Inference logic (if used)
├── requirements.txt
├── Dockerfile
└── openenv.yaml
```

---

## ⚙️ Installation

### 🔹 1. Clone the repository

```
git clone https://github.com/your-username/Openenv-Customer-Support.git
cd Openenv-Customer-Support
```

### 🔹 2. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Running the Server

```
python server.py
```

Server will start at:

```
http://localhost:7860
```

API docs available at:

```
http://localhost:7860/docs
```

---

## 📡 API Endpoints

### 🔹 Health Check

```
GET /
```

---

### 🔹 List Tasks

```
GET /tasks
```

---

### 🔹 Reset Environment

```
POST /reset?task=easy_classification
```

---

### 🔹 Take a Step

```
POST /step
```

Request body example:

```json
{
  "action_type": "classify",
  "value": "billing"
}
```

---

### 🔹 Get Current State

```
GET /state
```

---

### 🔹 Close Environment

```
POST /close
```

---

## 🧠 How It Works

1. Call `/reset` to start a new episode
2. Use `/step` to interact with the environment
3. Receive reward, state, and done flag
4. Repeat until the task is completed

---

## 🐳 Docker (Optional)

Build image:

```
docker build -t openenv-support .
```

Run container:

```
docker run -p 7860:7860 openenv-support
```

---

## 📌 Use Cases

* Reinforcement Learning environments
* AI agent evaluation
* Customer support automation testing
* Simulation-based training systems

---

## 🤝 Contributing

Feel free to fork the repo and submit pull requests.

---

## 📄 License

This project is open-source and available under the MIT License.
