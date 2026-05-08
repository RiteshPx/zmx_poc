
"""
AWS Bedrock — Standard Claude Connection + Document Extraction Template

Project Use Case:
- PDF understanding
- OCR text analysis
- Structured information extraction
- Invoice / form / legal document processing

Recommended Flow:
PDF/Image
    ↓
PyMuPDF + PaddleOCR
    ↓
Extracted Text
    ↓
Claude on Bedrock
    ↓
Structured JSON Output
    ↓
Coordinate Matching + Highlighting
"""

import json
import boto3
from botocore.exceptions import ClientError


# =========================================================
# CONFIGURATION
# =========================================================

REGION = "us-east-1"

# Claude Opus 4.5 inference profile
MODEL_ID = "us.anthropic.claude-opus-4-5-20251101-v1:0"

# Generation settings
MAX_TOKENS = 500
TEMPERATURE = 0


# =========================================================
# BEDROCK CLIENT
# =========================================================

def create_bedrock_client():
    """
    Create AWS Bedrock Runtime client.
    AWS credentials are automatically loaded from:
    - aws configure
    - environment variables
    - IAM role
    """

    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=REGION
    )

    return client


# =========================================================
# PROMPT BUILDER
# =========================================================

def build_extraction_prompt(document_text: str) -> str:
    """
    Create a professional extraction prompt.

    The model should:
    - understand OCR noise
    - extract important entities
    - return ONLY valid JSON
    """

    prompt = f"""
You are an intelligent document extraction system.

Your task is to analyze the provided document text and extract important business fields.

Instructions:
- Return ONLY valid JSON
- Do NOT add explanations
- Do NOT add markdown
- If a field is missing, return null
- Preserve original values exactly as written

Required Fields:
1. invoice_number
2. invoice_date
3. vendor_name
4. total_amount

Document Text:
-----------------------
{document_text}
-----------------------
"""

    return prompt


# =========================================================
# MODEL INVOCATION
# =========================================================

def invoke_claude(client, prompt: str):
    """
    Send request to Claude model via Bedrock.
    """

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response["body"].read())

    return response_body


# =========================================================
# RESPONSE PARSER
# =========================================================

def extract_text_response(response_body):
    """
    Extract Claude text response safely.
    """

    return response_body["content"][0]["text"]


# =========================================================
# MAIN TEST FUNCTION
# =========================================================

def main():

    print("=" * 70)
    print("AWS Bedrock — Claude Document Extraction Test")
    print("=" * 70)

    # -----------------------------------------------------
    # Example OCR / PDF text
    # Later this will come from:
    # - PyMuPDF
    # - PaddleOCR
    # -----------------------------------------------------

    sample_document = """
    Invoice Number: INV-101
    Invoice Date: 05/07/2026
    Vendor Name: ABC Technologies Pvt Ltd
    Total Amount: ₹54,200
    """

    try:

        # -------------------------------------------------
        # Create client
        # -------------------------------------------------

        client = create_bedrock_client()

        print(f"\n✅ Bedrock client connected")
        print(f"✅ Region      : {REGION}")
        print(f"✅ Model       : {MODEL_ID}")

        # -------------------------------------------------
        # Build prompt
        # -------------------------------------------------

        prompt = build_extraction_prompt(sample_document)

        # -------------------------------------------------
        # Invoke model
        # -------------------------------------------------

        print("\n⏳ Sending request to Claude...\n")

        response_body = invoke_claude(client, prompt)

        # -------------------------------------------------
        # Parse output
        # -------------------------------------------------

        output_text = extract_text_response(response_body)

        # -------------------------------------------------
        # Print results
        # -------------------------------------------------

        print("=" * 70)
        print("MODEL RESPONSE")
        print("=" * 70)

        print(output_text)

        # -------------------------------------------------
        # Usage stats
        # -------------------------------------------------

        usage = response_body.get("usage", {})

        print("\n" + "=" * 70)
        print("TOKEN USAGE")
        print("=" * 70)

        print(f"Input Tokens  : {usage.get('input_tokens')}")
        print(f"Output Tokens : {usage.get('output_tokens')}")

        print("\n🎉 Claude extraction pipeline working successfully.")

    except ClientError as error:

        error_code = error.response["Error"]["Code"]
        error_message = error.response["Error"]["Message"]

        print("\n❌ AWS Client Error")
        print(f"Code    : {error_code}")
        print(f"Message : {error_message}")

        if error_code == "AccessDeniedException":

            print("\nPossible Fixes:")
            print("1. Enable model access in Bedrock")
            print("2. Verify AWS credentials")
            print("3. Check Bedrock region")
            print("4. Ensure inference profile access")

    except Exception as error:

        print("\n❌ Unexpected Error")
        print(error)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()