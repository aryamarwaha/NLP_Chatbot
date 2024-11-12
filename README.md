# NLP Chatbot for Movie Recommendations 🎥

This project is an NLP-based chatbot developed for a CS124 assignment. It provides **movie recommendations** and performs **real-time sentiment analysis** on user input. The bot uses both rule-based NLP techniques and **Large Language Models (LLMs)** to enhance its conversational abilities, specifically for movie recommendation and sentiment extraction.

## Features
- **Movie Recommendations**: The chatbot collects user feedback on movies they like or dislike and recommends similar movies based on collaborative filtering.
- **Sentiment Analysis**: Extracts positive or negative sentiments toward movies mentioned by users to tailor responses.
- **LLM Integration**: In advanced modes, the bot leverages LLMs for more nuanced responses and focused interaction.

## Files and Structure
- **`chatbot.py`**: Core logic for processing input, extracting movie titles, analyzing sentiment, and generating responses.
- **`repl.py`**: Runs the chatbot in a Read-Eval-Print Loop (REPL) environment for user interaction.
- **Data and Utilities**: The `data` folder contains a movie database and sentiment lexicon. Helper scripts handle movie title extraction and collaborative filtering.

## Modes of Operation
1. **Starter Mode**: Basic mode with predefined NLP functions for title and sentiment extraction.
2. **LLM Prompting Mode**: Enhances recommendations with LLMs, ensuring the bot remains focused on movies.
3. **LLM Programming Mode**: Supports advanced features with structured JSON responses or customized prompts for improved interaction.

## Setup
1. Clone the repository.
   ```bash
   git clone https://github.com/aryamarwaha/NLP_Chatbot.git
Install dependencies and activate the environment:
conda activate cs124
pip install openai

Run the chatbot: python3 repl.py

Contributions
Developed by Arya Marwaha. Each team member contributed equally to the project.
