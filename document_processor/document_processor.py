import os
import json
import logging
from google import genai
from google.genai import types
from typing import Dict, Any
from ollama import chat
from ollama import ChatResponse
import pathlib
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("extraction.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes documents using AI for structured data extraction."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize document processor.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)

        logger.info("Document Processor initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(
                f"Configuration file not found at {config_path}. Using default configuration."
            )
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "critical_fields": ["quantity", "SKU", "description"],
            "confidence_threshold": 0.7,
            "manufacturer_list": [],
            "extraction_patterns": {
                "quantity": r"\b(\d+(?:\.\d+)?)\s*(?:ea|pcs|units|pieces)?\b",
                "SKU": r"\b([A-Z0-9\-\.]{5,20})\b",
                "unit_of_measure": r"\b(ea|pcs|units|pieces|box|boxes)\b",
            },
            "llm": {
                "enabled": True,
                "model": "llama3",
                "temperature": 0.1,
                "max_tokens": 500,
            },
        }

    def _get_hardware_estimation_prompt(self, prompt_file: str = "../prompts/hardware_estimation_prompt.txt") -> str:
        """Load hardware estimation prompt from file.

        Args:
            prompt_file: Path to prompt file

        Returns:
            Prompt text
        """
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            logger.info(f"Prompt loaded from {prompt_file}")
            return prompt
        except FileNotFoundError:
            logger.warning(f"Prompt file not found at {prompt_file}.")
            return self._fallback_prompt()

    def _fallback_prompt(self) -> str:
        """Return fallback prompt when file not found.

        Returns:
            Basic fallback prompt
        """
        return """Please analyze this hardware estimate document and extract all relevant information as structured JSON.
        
        Include:
        - Document info (customer, job number, date, vendor)
        - Hardware items (part number, quantity, unit of measure, description, manufacturer, prices)
        - Totals (subtotal, tax, total)
        
        Return only valid JSON."""

    def gemini_call(self, prompt: str) -> str:
        """Send prompt to Gemini API.

        Args:
            prompt: Text prompt

        Returns:
            Generated response
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        return response.text

    def ollama_call(self, prompt: str) -> str:
        """Send prompt to Ollama API.

        Args:
            prompt: Text prompt

        Returns:
            Generated response
        """
        response: ChatResponse = chat(
            model="gemma3:12b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response.message.content

    def process_document(
        self, document_path: str, custom_prompt: str = None
    ) -> Dict[str, Any]:
        """Process document using Gemini AI.

        Args:
            document_path: Path to document file
            custom_prompt: Optional custom prompt

        Returns:
            Extracted document data
        """
        filepath = pathlib.Path(document_path)
        client = genai.Client(api_key=self.config["GEMINI_API_KEY"])

        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self._get_hardware_estimation_prompt()

        mime_type = "application/pdf"
        if filepath.suffix.lower() in [".docx", ".doc"]:
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif filepath.suffix.lower() == ".txt":
            mime_type = "text/plain"

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(
                    data=filepath.read_bytes(),
                    mime_type=mime_type,
                ),
                prompt,
            ],
        )
        return response.text


def save_results(result: Dict[str, Any], output_file: str) -> None:
    """Save results to JSON file.

    Args:
        result: Data to save
        output_file: Output file path
    """
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    extractor = DocumentProcessor("../config.json")
    result = extractor.process_document("../docs/PurchaseOrderTemplate.pdf")
    print(result)
    save_results(result, "../docs/PurchaseOrderTemplate.json")
