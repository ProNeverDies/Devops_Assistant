#  DevOps Assistant (LLM-Powered) — Windows Edition

> A smart, terminal-based assistant to monitor, audit, and automate your DevOps workflows  powered by Ollama, LangChain, and Python.


##  Features

-  **Interactive Chat Interface** (CLI) with local LLM
-  LLM-backed responses (Ollama + Qwen/Orca/Phi3)
-  Docker monitoring and status summaries
-  Pytest integration with coverage
-  Security auditing via pip-audit
-  Real-time **System Metrics & Health**
-  IMAP Email Monitoring + Summarization
-  Git status analysis (on any repo)
-  Alert webhook integration (Slack/WebhookRelay/Teams/etc.)
-  Periodic background monitoring
-  YAML-based configuration
-  Works fully offline (once models are downloaded)

  ##  Installation

###  Prerequisites

- Windows 10/11 (64-bit)
- Python 3.10+ (Add to PATH)
- [Ollama](https://ollama.ai) installed (`phi3`, `orca-mini`, or `qwen`)
- Git + PowerShell
- Optional: Docker Desktop

---

###  Install Python Packages

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
