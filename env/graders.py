import re
from typing import Any, Dict, Tuple
from env.models import Action
COURTESY_TERMS = {
    "standard": [
        "sorry", "apologize", "apology", "apologies", "regret", "sincerely",
        "understand", "appreciate", "acknowledge", "hear you", "frustrat",
        "inconvenience", "concern", "difficult", "must be", "thank you", 
        "thanks", "grateful", "happy to help", "glad to assist", "here to help", 
        "assist you", "please", "let us know", "reach out", "feel free", 
        "contact us", "assure", "we value", "important to us", "committed", 
        "priority", "resolve", "ensure", "take care"
    ],
    "empathy": [
        "i understand how", "i can imagine", "i know how frustrat",
        "that must be", "completely understand", "truly sorry",
        "deeply apologize", "i sincerely", "we sincerely",
        "this is not the experience", "you deserve", "we value you"
    ],
    "hostile": [
        "not our problem", "your fault", "impossible", "cannot help",
        "don't care", "too bad", "get over it", "not responsible",
        "nothing we can do"
    ],
    "automated": [
        "as an ai", "i am an ai", "as a language model",
        "i don't have access", "i cannot access"
    ]
}
FINANCIAL_LEXICON = {
    "actions": ["initiat", "process", "submit", "approv", "complet", "issu", "creat", "raised", "opened", "filed"],
    "timing": ["business day", "working day", "3-5", "5-7", "7-10", "within", "24 hour", "48 hour", "week", "shortly"],
    "subjects": ["refund", "reimburs", "credit", "return", "amount", "eligible", "policy", "payment", "charge", "transaction"]
}
URGENCY_MARKERS = {
    "critical": [
        "damaged", "defective", "broken", "cracked", "scratched", "destroyed",
        "ruined", "faulty", "malfunction", "rude", "unprofessional", "dismissive", 
        "hung up", "disrespect", "terrible", "horrible", "worst", "awful", 
        "unacceptable", "outrageous", "disgusting", "repeated", "again", 
        "third time", "fourth time", "multiple times", "every time", "keep", 
        "always", "manager", "supervisor", "legal", "lawsuit", "sue", "court",
        "news", "social media", "twitter", "facebook", "review", "unsafe", 
        "dangerous", "hazard", "chemical", "smell"
    ],
    "moderate": [
        "frustrated", "disappointed", "upset", "unhappy", "not happy",
        "concerned", "worried", "anxious", "lost", "missing"
    ]
}
def evaluate_intent_alignment(action: Action, target_label: str) -> Tuple[float, str]:
    if not action.predicted_category:
        return 0.0, "Missing prediction"
    label_pred = action.predicted_category.lower().strip()
    label_true = target_label.lower().strip()
    if label_pred == label_true:
        return 1.0, f"Match: {label_pred}"
    semantic_clusters = [
        {"delivery", "shipping", "shipment", "dispatch"},
        {"refund", "return", "exchange", "reimbursement"},
        {"complaint", "feedback", "issue", "problem", "concern"}
    ]
    for cluster in semantic_clusters:
        if label_pred in cluster and label_true in cluster:
            return 0.5, f"Partial Match: {label_pred}/{label_true}"      
    return 0.0, f"Mismatch: {label_pred} vs {label_true}"
def evaluate_sentiment_tone(action: Action) -> Tuple[float, str]:
    content = action.response_text.lower()
    count_polite = sum(1 for w in COURTESY_TERMS["standard"] if w in content)
    count_empathy = sum(1 for w in COURTESY_TERMS["empathy"] if w in content)
    count_neg = sum(1 for w in COURTESY_TERMS["hostile"] if w in content)
    count_bot = sum(1 for w in COURTESY_TERMS["automated"] if w in content)
    base_val = min(count_polite / 4.0, 1.0)
    bonus_val = min(count_empathy * 0.1, 0.2)
    deduction = (count_neg * 0.3) + (count_bot * 0.15)
    final_score = max(0.0, min(1.0, base_val + bonus_val - deduction))
    log = f"p={count_polite}, e={count_empathy}, n={count_neg}, f={count_bot}"
    return round(final_score, 4), log
def evaluate_prose_quality(action: Action) -> Tuple[float, str]:
    raw_text = action.response_text.strip()
    clean_text = raw_text.lower()
    words = raw_text.split()
    w_len = len(words)
    s_len = max(1, len([s for s in re.split(r"[.!?]+", raw_text) if s.strip()]))
    points = 0.0
    tags = []
    if 30 <= w_len <= 150:
        points += 0.35
        tags.append(f"ideal_len({w_len})")
    elif w_len > 150:
        points += 0.25
        tags.append(f"long({w_len})")
    elif w_len >= 15:
        points += 0.15
        tags.append(f"brief({w_len})")
    else:
        tags.append(f"insufficient({w_len})")
    if s_len >= 3:
        points += 0.25
        tags.append("strong_structure")
    elif s_len >= 2:
        points += 0.15
        tags.append("basic_structure")
    cta_regex = r"(feel free|contact us|reach out|let us know|here to help|please (reply|respond|call|email)|best regards|sincerely|thank you for|do not hesitate|any (other|further|additional) (question|concern))"
    if re.search(cta_regex, clean_text):
        points += 0.25
        tags.append("cta_present")
    context_keywords = [r"\border\b", r"\bticket\b", r"\bcase\b", r"\bitem\b", r"\bpackage\b", r"\bshipment\b", r"\brefund\b", r"\baccount\b", r"\binvestigat\b", r"\breview\b", r"\bcheck\b"]
    context_hits = sum(1 for p in context_keywords if re.search(p, clean_text))
    if context_hits >= 3:
        points += 0.15
        tags.append(f"high_context({context_hits})")
    elif context_hits >= 1:
        points += 0.05
        tags.append(f"low_context({context_hits})")
    return round(min(points, 1.0), 4), ", ".join(tags)
def evaluate_refund_details(action: Action) -> Tuple[float, str]:
    body = action.response_text.lower()
    hits_key = sum(1 for x in FINANCIAL_LEXICON["subjects"] if x in body)
    hits_act = sum(1 for x in FINANCIAL_LEXICON["actions"] if x in body)
    hits_time = sum(1 for x in FINANCIAL_LEXICON["timing"] if x in body)
    points = 0.0
    logs = []
    if hits_key >= 2:
        points += 0.4
        logs.append(f"keys:{hits_key}")
    elif hits_key == 1:
        points += 0.2
        logs.append("key:1")
    if hits_act >= 1:
        points += 0.3
        logs.append(f"acts:{hits_act}")
    if hits_time >= 1:
        points += 0.3
        logs.append(f"time:{hits_time}") 
    return round(min(points, 1.0), 4), ", ".join(logs) or "missing_info"
def evaluate_escalation_logic(action: Action, raw_ticket: str, rank: str, mood: str) -> Tuple[float, str]:
    src = raw_ticket.lower()
    reply = action.response_text.lower()
    has_hard = any(i in src for i in URGENCY_MARKERS["critical"])
    has_soft = any(i in src for i in URGENCY_MARKERS["moderate"])
    is_high = rank == "high"
    is_neg = mood == "negative"
    did_escalate = bool(action.should_escalate)
    force_escalate = has_hard or (is_high and is_neg)
    maybe_escalate = has_soft and (is_high or is_neg)
    meta = f"H:{has_hard} S:{has_soft} P:{rank} M:{mood} E:{did_escalate}"
    if force_escalate:
        if did_escalate:
            patterns = [r"escalat", r"senior", r"specialist", r"supervisor", r"manager", r"team", r"priorit", r"urgent"]
            valid_reply = any(re.search(p, reply) for p in patterns)
            return round(min(1.0, 0.9 + (0.1 if valid_reply else 0.0)), 4), f"Valid Escalate. {meta}"
        return 0.0, f"Missed Mandatory Escalate. {meta}"
    if maybe_escalate:
        return (0.7, f"Borderline Escalate. {meta}") if did_escalate else (0.6, f"Borderline Hold. {meta}")
    if not did_escalate:
        return 1.0, f"Correct Hold. {meta}"
    return 0.3, f"Unnecessary Escalate. {meta}"
def grade(action: Action, task_name: str, ticket: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    report = {}
    s_cat, n_cat = evaluate_intent_alignment(action, ticket["category"])
    report["category_match"] = {"score": s_cat, "note": n_cat}
    s_pol, n_pol = evaluate_sentiment_tone(action)
    report["politeness"] = {"score": s_pol, "note": n_pol}
    s_qual, n_qual = evaluate_prose_quality(action)
    report["response_quality"] = {"score": s_qual, "note": n_qual}
    if task_name == "easy_classification":
        final = (s_cat * 0.45) + (s_pol * 0.25) + (s_qual * 0.30)
    elif task_name == "refund_handling":
        s_ref, n_ref = evaluate_refund_details(action)
        report["refund_clarity"] = {"score": s_ref, "note": n_ref}
        final = (s_cat * 0.20) + (s_pol * 0.20) + (s_qual * 0.25) + (s_ref * 0.35)
    elif task_name == "escalation_decision":
        s_esc, n_esc = evaluate_escalation_logic(action, ticket["text"], ticket.get("priority", "medium"), ticket.get("sentiment", "neutral"))
        report["escalation_accuracy"] = {"score": s_esc, "note": n_esc}
        final = (s_cat * 0.15) + (s_pol * 0.20) + (s_qual * 0.20) + (s_esc * 0.45)
    else:
        final = (s_cat + s_pol + s_qual) / 3.0
    res = round(final, 4)
    report.update({"final_reward": res, "task": task_name, "version": "v2-clean"})
    return res, report