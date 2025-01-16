# 🌟 **Function Calling Benchmark for MPC Conversations** 🌟

Welcome to the **MPC Function Calling Benchmark**! This project is designed to benchmark function calling in **multi-party conversational (MPC)** scenarios. Dive in to explore agents, functions, and workflows in a seamless environment. 💬📊

---

## ⚙️ **Environment Setup**

To get started, ensure your environment is set up correctly:

- **🐍 Python Version:** 3.11.10
- **📦 Dependencies and Conda Environment:** Refer to `requirements.txt` and `environment.yml`

### 🛠️ **Install Requirements**
Run the following command to install dependencies:
```bash
pip install -r requirements.txt
```

To create a conda enironment and install dependencies:
```bash
conda env create -f environment.yml
```

---

## 🚀 **How to Run**

Run the application using:
```bash
python main.py
```

### 🔧 **Custom Configuration**
You can customize configurations by editing `src_configs/gen_base.yaml`. Use the following command to specify your configuration file:
```bash
python main.py --yaml_path=src_configs/gen_base.yaml
```

---

## 🌐 **Streamlit Interface**

For a visual and interactive experience, launch the Streamlit dashboard:
```bash
streamlit run showing_data.py
```

> considering to use `taipy` instead of streamlit

---

## 🏗️ **Project Structure**

Here’s an overview of the project structure:

- **🤖 `agent/`**: Implements agent nodes and builds the LangChain pipeline.
- **📋 `function/`**: Stores and objectifies the function list.
- **📊 `graph/`**: (Coming Soon) Graph-related modules for advanced workflows.
- **📓 `notebook/`**: Jupyter notebooks for analysis and prototyping.
- **💾 `outputs/`**: Stores output data.
- **💡 `prompt/`**: Contains prompt templates and configurations.
- **🛠️ `utils/`**: Utility scripts for file operations and helper functions.
- **🚦 `main.py`**: The main entry point for the application.
- **🧠 `inference.py`**: Manages inference logic (needs updates for `vllm` compatibility).

---

## 🤝 **Contributing**

We’re excited to see your contributions! Here’s how you can help:
- **📊 Graph Modules**: Add graph implementations under `graph/`.
- **🛠️ Update `inference.py`**: Improve compatibility with the `vllm` library.

---

## 💡 **Final Notes**
Feel free to modify this project as it evolves. We appreciate your feedback and contributions to make this benchmark even better. 🌟✨
