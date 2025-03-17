# Agentic Retrieval-Augmented Generation (RAG) Project

This project implements a sophisticated Retrieval-Augmented Generation (RAG) system with agentic capabilities, designed for robust and efficient conversational AI.

## Project Description

This repository contains a RAG implementation that combines various advanced techniques to provide accurate and context-aware responses. It leverages:

* **Documents Info:**
    * Exploration of attention mechanisms.
    * Usage of cutting-edge models like GPT-4, Gemini, Mistral-7B, and InstructGPT.
* **LangGraph:**
    * Implementation of complex workflows and agentic behavior using LangGraph.
* **Agentic RAG:**
    * Integration of web search capabilities to enhance information retrieval.
* **Conversation History Preservation:**
    * Maintains conversation context to provide coherent and relevant responses.
* **Streamlit Application:**
    * User-friendly interface for interacting with the RAG system.
* **Caching Capability:**
    * Implements caching to improve response generation speed and reduce retrieval overhead.
* **Context Window Management:**
    * Provides control over the number of historical messages included in the context window.
    * Limits the history to a maximum of the 5 latest question-and-answer pairs to meet context window requirements.

## Key Features

* **Advanced RAG Pipeline:** Combines retrieval and generation for accurate and context-aware responses.
* **Agentic Capabilities:** Integrates web search for enhanced information retrieval.
* **Conversational Memory:** Preserves conversation history for coherent interactions.
* **Efficient Response Generation:** Implements caching to improve speed and reduce resource usage.
* **Context Window Control:** Manages historical context to fit within model limitations.
* **User-Friendly Interface:** Streamlit app for easy interaction.
* **ChatGPT-like streaming capability:** Answers appear word by word, providing a more interactive and engaging user experience.
