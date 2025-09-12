
import os
from typing import Dict, List
from dotenv import load_dotenv
from agent.classifier_agent import classify_ticket
from rag import retrieval

load_dotenv()

class EscalationDecisionEngine:
	def __init__(self, escalation_threshold: float = None):
		# Read from env if not provided
		if escalation_threshold is None:
			escalation_threshold = float(os.getenv("ESCALATION_THRESHOLD", 0.6))
		self.escalation_threshold = escalation_threshold

	def should_escalate(self, query: str, classification: Dict, similarity_scores: List[float], retrieved_docs: List[Dict], draft_answer: str = None) -> Dict:
		"""
		Decide whether to escalate based on retrieval similarity, classifier, and answer quality.
		Returns dict with should_escalate, escalation_score, factors, reasoning.
		"""
		# --- Confidence from retrieval ---
		max_sim = max(similarity_scores) if similarity_scores else 0.0
		avg_sim = sum(similarity_scores)/len(similarity_scores) if similarity_scores else 0.0
		num_good = sum(1 for s in similarity_scores if s > 0.6)
		retrieval_confidence = (0.5 * max_sim) + (0.3 * avg_sim) + (0.2 * (num_good/5))

		# --- Query complexity ---
		complexity = 0.0
		if len(query.split('?')) > 1:
			complexity += 0.3
		if len(query.split()) > 50:
			complexity += 0.2
		if any(word in query.lower() for word in ['integrate', 'custom', 'setup', 'troubleshoot']):
			complexity += 0.2

		# --- Sentiment/urgency ---
		sentiment = classification.get('sentiment', 'Neutral')
		priority = classification.get('priority', 'P2')
		sentiment_urgency = 0.0
		if sentiment == 'Angry':
			sentiment_urgency += 0.9
		elif sentiment == 'Frustrated':
			sentiment_urgency += 0.7
		if priority == 'P0':
			sentiment_urgency += 0.8
		elif priority == 'P1':
			sentiment_urgency += 0.4

		# --- Topic criticality ---
		topic = classification.get('topic', 'Unknown')
		critical_topics = {
			'Sensitive data': 0.9,
			'Security': 0.8,
			'Billing': 0.7,
			'Compliance': 0.8,
			'Integration': 0.6,
			'Custom': 0.7
		}
		topic_criticality = critical_topics.get(topic, 0.2)

		# --- Response quality (if draft answer provided) ---
		response_quality = 0.0
		if draft_answer:
			if len(draft_answer.split()) < 20:
				response_quality += 0.2
			if 'no relevant information' in draft_answer.lower():
				response_quality += 0.4

		# --- Weighted escalation score ---
		weights = {
			'retrieval_confidence': 0.3,
			'complexity': 0.2,
			'sentiment_urgency': 0.2,
			'topic_criticality': 0.2,
			'response_quality': 0.1
		}
		factors = {
			'retrieval_confidence': 1 - retrieval_confidence,  # low conf = escalate
			'complexity': complexity,
			'sentiment_urgency': sentiment_urgency,
			'topic_criticality': topic_criticality,
			'response_quality': response_quality
		}
		escalation_score = sum(factors[k] * weights[k] for k in weights)
		should_escalate = escalation_score > self.escalation_threshold
		reasoning = []
		if factors['retrieval_confidence'] > 0.5:
			reasoning.append('Low retrieval confidence')
		if factors['complexity'] > 0.3:
			reasoning.append('Complex query')
		if factors['sentiment_urgency'] > 0.7:
			reasoning.append('High urgency/sentiment')
		if factors['topic_criticality'] > 0.7:
			reasoning.append('Critical topic')
		if factors['response_quality'] > 0.3:
			reasoning.append('Low answer quality')
		return {
			'should_escalate': should_escalate,
			'escalation_score': escalation_score,
			'factors': factors,
			'reasoning': reasoning
		}

class RAGAgent:
	def __init__(self, min_score: float = 0.1, max_docs: int = None, escalation_threshold: float = None):
		# Read from env if not provided
		if max_docs is None:
			max_docs = int(os.getenv("RAG_MAX_DOCS", 5))
		self.max_docs = max_docs
		self.min_score = min_score
		self.escalation_engine = EscalationDecisionEngine(escalation_threshold)

	def process_query(self, query: str, min_score: float = None, max_docs: int = None, escalation_threshold: float = None) -> Dict:
		"""
		1. Retrieve all relevant docs (with scores) above min_score.
		2. Classify query.
		3. Run escalation logic.
		4. If escalate: return escalation response. Else: generate answer using same docs.
		"""
		# Allow override of min_score, max_docs, and escalation_threshold at call time
		min_score = min_score if min_score is not None else self.min_score
		max_docs = max_docs if max_docs is not None else self.max_docs
		if escalation_threshold is not None:
			self.escalation_engine.escalation_threshold = escalation_threshold

		# 1. Retrieve all relevant docs with scores (only once)
		doc_score_pairs = retrieval.retrieve_relevant_docs_with_scores(query, min_score=min_score, max_docs=max_docs)
		retrieved_docs = [doc for doc, score in doc_score_pairs]
		similarity_scores = [score for doc, score in doc_score_pairs]

		# 2. Classify query
		classification = classify_ticket(query)

		# 3. Optionally, generate a draft answer for quality check
		context = retrieval.format_context_from_docs(retrieved_docs)
		draft_answer = None
		if context:
			formatted_prompt = retrieval.assistant_prompt.format(context=context, query=query)
			draft_answer = retrieval.llm.invoke(formatted_prompt).content.strip()

		# 4. Escalation logic
		escalation_decision = self.escalation_engine.should_escalate(
			query, classification, similarity_scores, retrieved_docs, draft_answer
		)
		if escalation_decision['should_escalate']:
			return {
				'type': 'escalation',
				'content': (
					f"Your query requires attention from a human support agent. "
					f"Reasons: {', '.join(escalation_decision['reasoning']) or 'Threshold exceeded'} "
					f"(Score: {escalation_decision['escalation_score']:.2f})"
				),
				'classification': classification,
				'escalation_score': escalation_decision['escalation_score'],
				'factors': escalation_decision['factors'],
				'reasoning': escalation_decision['reasoning'],
				'sources': []
			}
		# 5. Not escalated: generate final answer using same docs/context
		if context:
			formatted_prompt = retrieval.assistant_prompt.format(context=context, query=query)
			final_answer = retrieval.llm.invoke(formatted_prompt).content.strip()
		else:
			final_answer = "No relevant information found in the documentation. This ticket should be routed to the support team."
		return {
			'type': 'ai_response',
			'content': final_answer,
			'classification': classification,
			'escalation_score': escalation_decision['escalation_score'],
			'factors': escalation_decision['factors'],
			'reasoning': escalation_decision['reasoning'],
			'sources': []  # Optionally parse sources from answer
		}

# Example usage
if __name__ == "__main__":
	agent = RAGAgent()
	test_queries = [
		"How do I set up SSO authentication with SAML?",
		"I'm really frustrated! Your billing system charged me twice!",
		"Can you help me with custom integration for our database?"
	]
	for q in test_queries:
		print(f"\nQuery: {q}")
		# Example: override max_docs and threshold at call time
		result = agent.process_query(q, max_docs=7, escalation_threshold=0.7)
		print("Type:", result['type'])
		print("Content:", result['content'])
		print("Classification:", result['classification'])
		print("Escalation Score:", result['escalation_score'])
		print("Factors:", result['factors'])
		print("Reasoning:", result['reasoning'])
