# Free Document Property Extractor
_Automated Document Property Extraction._

A Python script for extracting properties from a batch of PDF files. This tool is designed for legal professionals who need to efficiently catalog and manage large volumes of discovery documents.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## Overview

This script builds on the output from the *Free_CLassifier* project. For each document in the output file created by *Free_Classifier*, this script will extract certain document properties based on the document's classification. Finally a list of docuoments and their properties is saved to a CSV/Excel file.

## Features

- **Multimodal Input Support**: Processes PDF files that are either scanned images or searchable. It will also process common image formats such as JPG, GIF, and PNG.
- **Hidden Directory Filtering**: Automatically excludes directories starting with "." (following Linux convention)
- **Error Handling**: Continues processing even when individual files encounter issues
- **CSV Output**: Generates structured data ready for import into legal case management systems

## Built With

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/-OpenAI-412991?logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/-Anthropic-D97757?logo=claude&logoColor=white)
![Gemini](https://img.shields.io/badge/-Gemini-8E75B2?logo=googlegemini&logoColor=white)

## Requirements

- Python 3.6+
- Git client
- openai library
- anthropic library
- googlegemini library
- *See also*, ```requirements.txt```

## Prerequisites

*You will need to have the ```git``` software installed on your computer before you begin ([GIT Installation Page](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)). For Windows, you can install from this link: [Windows Installation](https://git-scm.com/download/win). For Mac, you can install from this link: [Mac Installation](https://git-scm.com/download/mac).*

*You need the ```python``` interpreter installed on your system before you can run the script. ([Python Installation page](https://www.python.org/downloads/windows/)).*

*You need an API token from one or more of the following: OpenAI, Anthropic, Google Gemini*

## Installation

1. Clone this repository:
```bash
git clone https://github.com/tjdaley/free_property_extractor.git
cd free_property_extractor
```

2. Create a virtual environment and activate the environment:
```bash
python -m venv venv
venv\scripts\activate.bat
```

4. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Update the .env File

The ```.env``` file controls certain aspects of how the program works. Here are the variables you need to set:

Variable | Description | Permitted Values
----|----|----
llm_name | The LLM vendor you are using | openai or anthropic or gemini
llm_model | The LLM model you are using | It must be a multimodal model (one that will process text and images). The LLM vendors list their models on their web sites.
llm_api_key | The API key assigned to you by the LLM vendor | Any string provided by the vendor
prompt_file | Name of the file that contains the classification prompt | Any valid filename
input_csv | Path to the output file from *free_classifier* | Any valid filename
output_csv | Path to where this script will write its output | Any valid filename
mapping_file | Mapping of document classifications/labels to propoerties to be extracted | Any valid filename
max_pages | Maximum number of pages to submit to the LLM for processing | Positive integer number > 0

### Basic Usage

```bash
python property_extractor.py
```

## Output Format

The script generates a CSV file with the following columns:
- `filename`: Name of the PDF file (without path)
- `path`: Full path name of the file
- `label`: The Classification label for the document
- `etc` : List of properties for the document, based on the label

## Error Handling

The script includes comprehensive error handling:
- Skips non-existent files with warnings
- Continues processing if individual PDFs encounter errors
- Reports processing status for each file
- Logs errors without stopping the entire process

## Advanced Features

### Hidden Directory Filtering
When scanning directories, the script automatically excludes any directories starting with "." to avoid processing hidden system folders.

## Troubleshooting


## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Thomas J. Daley** is a family law litigation attorney practicing primarily in Collin County, Texas and representing clients in family disputes throughout the State of Texas and the United States. As a tech entrepreneur, he leverages AI to bring high-quality legal services that work better, faster, and cheaper than traditional approaches to resolving cases.

---

<p align="center">Made with ❤️ in Texas by <a href="https://github.com/tjdaley">Tom Daley</a></p>