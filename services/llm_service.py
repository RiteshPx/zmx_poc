"""
services/llm_service.py

Sends extracted PDF text to Claude on AWS Bedrock.
Returns structured JSON with extracted fields.
"""

import json
import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

REGION   = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-opus-4-5-20251101-v1:0")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1000))


def _get_client():
    return boto3.client("bedrock-runtime", region_name=REGION)


def _build_prompt(document_text: str, doc_type: str = "auto") -> str:
    """
    Build a structured extraction prompt.
    doc_type: "invoice" | "kyc" | "resume" | "bank_statement" | "auto"
    """

    field_sets = {
        "invoice": [
            "invoice_number", "invoice_date", "vendor_name",
            "customer_name", "total_amount", "tax_amount", "due_date"
        ],
        "kyc": [
            "full_name", "date_of_birth", "address",
            "id_number", "id_type", "issue_date", "expiry_date"
        ],
        "resume": [
            "full_name", "email", "phone",
            "current_role", "years_experience", "skills", "education"
        ],
        "bank_statement": [
            "account_holder", "account_number", "bank_name",
            "statement_period", "opening_balance", "closing_balance"
        ],
        "auto": [
            "name", "date", "id_number", "amount",
            "address", "phone", "email", "organization"
        ]
    }

    fields = field_sets.get(doc_type, field_sets["auto"])
    fields_str = "\n".join(f"- {f}" for f in fields)

    return f"""You are an intelligent document extraction system.

Analyze the document text below and extract the key fields.

Rules:
- Return ONLY valid JSON, no explanation, no markdown, no code blocks
- If a field is not found, set it to null
- Preserve original values exactly as written (keep ₹ symbols, dashes, etc.)
- For dates, keep the original format found in the document

Fields to extract:
{fields_str}

Document Text:
---
{document_text}
---

Respond with JSON only."""


def extract_fields_with_llm(document_text: str, doc_type: str = "auto") -> dict:
    """
    Send document text to Claude and get structured extraction back.

    Returns:
        {
            "fields": {"invoice_number": "INV-101", ...},
            "usage": {"input_tokens": 450, "output_tokens": 80}
        }
    """
    client = _get_client()
    prompt = _build_prompt(document_text, doc_type)

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens"       : MAX_TOKENS,
        "temperature"      : 0,
        "messages"         : [{"role": "user", "content": prompt}]
    }

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )

        body  = json.loads(response["body"].read())
        text  = body["content"][0]["text"].strip()
        usage = body.get("usage", {})

        # Strip markdown fences if model accidentally adds them
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        fields = json.loads(text)
        return {"fields": fields, "usage": usage}

    except json.JSONDecodeError as e:
        return {"fields": {}, "usage": {}, "error": f"JSON parse error: {e}"}
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg  = e.response["Error"]["Message"]
        raise RuntimeError(f"Bedrock error [{code}]: {msg}")