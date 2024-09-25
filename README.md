# RAG Solution for Kol Zhot Israeli Civil Rights Website

This project is a Retrieval-Augmented Generation (RAG) solution designed to answer user questions using a Large Language Model (LLM) based on data from the Kol Zhot API. The goal is to provide accurate and relevant information regarding civil rights in Israel.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The RAG solution integrates a powerful LLM with the Kol Zhot API to deliver precise answers to user queries about civil rights in Israel. By leveraging the latest advancements in natural language processing, this project aims to enhance the accessibility and understanding of civil rights information.

## Features

- **Natural Language Understanding**: Utilizes a state-of-the-art LLM to comprehend and respond to user questions.
- **Data Integration**: Fetches relevant data from the Kol Zhot API to provide accurate answers.
- **User-Friendly Interface**: Designed to be intuitive and easy to use for individuals seeking information on civil rights.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/kol-zhot-rag-solution.git
    cd kol-zhot-rag-solution
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    Create a `.env` file in the root directory and add your API keys and other configuration settings.

## Usage

To start the application, run:

You can then access the application through your web browser at `http://localhost:5000`.

## Configuration

Ensure you have the following environment variables set in your `.env` file:

- `KOL_ZHOT_API_KEY`: Your API key for accessing the Kol Zhot API.
- `LLM_API_KEY`: Your API key for accessing the Large Language Model service.

## Contributing

We welcome contributions to enhance the project. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.
