import json
import re
from backend.llm_manager import create_llm

def parse_evaluation_response(text):
    """
    Robust reusable parsing logic to extract evaluation metrics from LLM output.
    Returns a structured dictionary.
    """
    result = {
        "correctness_score": 0,
        "confidence_score": 0,
        "hallucination_risk": "Medium",
        "completeness": "Medium",
        "verdict": "Unreliable",
        "detailed_reason": "Failed to parse evaluation metrics properly."
    }
    
    if not text:
        return result

    try:
        json_match = re.search(r'\{.*\}', text.strip(), re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            if "correctness_score" in parsed:
                result["correctness_score"] = int(parsed["correctness_score"])
            elif "Correctness Score" in parsed:
                result["correctness_score"] = int(parsed["Correctness Score"])
                
            if "confidence_score" in parsed:
                val = str(parsed["confidence_score"]).replace('%', '')
                result["confidence_score"] = int(float(val))
            elif "Confidence Score" in parsed:
                val = str(parsed["Confidence Score"]).replace('%', '')
                result["confidence_score"] = int(float(val))
                
            for key in ["hallucination_risk", "Hallucination Risk"]:
                if key in parsed:
                    val = str(parsed[key]).capitalize()
                    if any(x in val for x in ["Low", "Medium", "High"]):
                        result["hallucination_risk"] = [x for x in ["Low", "Medium", "High"] if x in val][0]
                    else:
                        result["hallucination_risk"] = val
                        
            for key in ["completeness", "Completeness"]:
                if key in parsed:
                    val = str(parsed[key]).capitalize()
                    if any(x in val for x in ["Low", "Medium", "High"]):
                        result["completeness"] = [x for x in ["Low", "Medium", "High"] if x in val][0]
                    else:
                        result["completeness"] = val
                        
            for key in ["verdict", "Verdict"]:
                if key in parsed:
                    val = str(parsed[key])
                    for v in ["Reliable", "Partially Reliable", "Unreliable"]:
                        if v.lower() in val.lower():
                            result["verdict"] = v
                            break
                    else:
                        result["verdict"] = val
                        
            for key in ["detailed_reason", "Detailed Reason", "reason", "Reason"]:
                if key in parsed:
                    result["detailed_reason"] = str(parsed[key]).strip()
                    break
            return result
    except Exception as e:
        print(f"[WARN] JSON parsing failed, falling back to regex parsing: {e}")

    try:
        corr_match = re.search(r'Correctness Score.*?(\d+)', text, re.IGNORECASE)
        if corr_match:
            result["correctness_score"] = int(corr_match.group(1))
            
        conf_match = re.search(r'Confidence Score.*?(\d+)', text, re.IGNORECASE)
        if conf_match:
            result["confidence_score"] = int(conf_match.group(1))
            
        hall_match = re.search(r'Hallucination Risk.*?(Low|Medium|High)', text, re.IGNORECASE)
        if hall_match:
            result["hallucination_risk"] = hall_match.group(1).capitalize()
            
        comp_match = re.search(r'Completeness.*?(Low|Medium|High)', text, re.IGNORECASE)
        if comp_match:
            result["completeness"] = comp_match.group(1).capitalize()
            
        verd_match = re.search(r'Verdict.*?(Reliable|Partially Reliable|Unreliable)', text, re.IGNORECASE)
        if verd_match:
            val = verd_match.group(1).lower()
            if "partially" in val:
                result["verdict"] = "Partially Reliable"
            elif "unreliable" in val:
                result["verdict"] = "Unreliable"
            else:
                result["verdict"] = "Reliable"
                
        reason_match = re.search(r'Detailed Reason.*?:(.*)', text, re.IGNORECASE | re.DOTALL)
        if reason_match:
            result["detailed_reason"] = reason_match.group(1).strip()
        else:
            reason_match2 = re.search(r'Reason.*?:(.*)', text, re.IGNORECASE | re.DOTALL)
            if reason_match2:
                result["detailed_reason"] = reason_match2.group(1).strip()
            else:
                result["detailed_reason"] = text.strip()
    except Exception as e:
        print(f"[ERROR] Fallback regex parsing error: {e}")
        result["detailed_reason"] = text.strip()

    return result

def evaluate_answer(context, answer):
    """
    Evaluates the generated answer against the retrieved context.
    """
    eval_prompt = f"""You are an advanced RAG evaluation system.
Evaluate the generated answer strictly based on the retrieved context.

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate:
1. Correctness Score (0-10)
2. Confidence Score (0-100%)
3. Hallucination Risk (Low / Medium / High)
4. Completeness (Low / Medium / High)
5. Verdict (Reliable / Partially Reliable / Unreliable)
6. Detailed Reason

Rules:
- High confidence only if answer is fully supported by context
- Detect hallucinations and unsupported claims
- Be strict and realistic
- Do not always give high scores
- Assign lower confidence for ambiguous or incomplete answers

Provide your output as a valid JSON object with EXACTLY the following keys:
{{
  "correctness_score": <int 0-10>,
  "confidence_score": <int 0-100>,
  "hallucination_risk": "<Low | Medium | High>",
  "completeness": "<Low | Medium | High>",
  "verdict": "<Reliable | Partially Reliable | Unreliable>",
  "detailed_reason": "<string explanation>"
}}
Do not include any markdown formatting or codeblocks outside the JSON object."""

    try:
        llm = create_llm()
        eval_response = llm.invoke(eval_prompt)
        return parse_evaluation_response(eval_response.content)
    except Exception as e:
        print(f"[ERROR] LLM Evaluation error: {e}")
        return {
            "correctness_score": 0,
            "confidence_score": 0,
            "hallucination_risk": "High",
            "completeness": "Low",
            "verdict": "Unreliable",
            "detailed_reason": f"Evaluation engine exception: {str(e)}"
        }
