"""
agents/study_agent.py

Main coordinator for UniStudy AI.
Controls action routing, tool execution, bounded iterations, and observable tracing.
"""

import json
import re
from typing import Dict, Any, List, Optional

from llm.ollama_client import OllamaClient
from tools.document_search import search_documents
from tools.summarizer import summarize_document
from tools.quiz_generator import generate_quiz
from tools.answer_evaluator import evaluate_answer
from utils.prompts import (
    ROUTER_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    format_qa_prompt
)


class StudyAgent:
    def __init__(self, client: OllamaClient, max_iterations: int = 3):
        self.client = client
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _record_trace(self, step: int, tool: str, input_summary: str, result_summary: str, status: str = "success") -> None:
        """Appends an observable, non-private audit record of a tool execution."""
        self.trace.append({
            "step": step,
            "tool": tool,
            "input_summary": input_summary,
            "result_summary": result_summary,
            "status": status
        })

    def route_intent(self, user_query: str) -> Dict[str, str]:
        """
        Determines the requested action using Ollama with deterministic rule fallback.

        Returns:
            dict: {"action": str, "search_query": str}
        """
        q_lower = user_query.lower().strip()

        # Deterministic overrides for unmistakable intents
        if any(w in q_lower for w in ["quiz", "test me", "mcq", "multiple choice", "practice question"]):
            return {"action": "generate_quiz", "search_query": user_query}
        if any(w in q_lower for w in ["summarize", "summary", "overview", "recap", "tl;dr"]):
            return {"action": "summarize", "search_query": user_query}
        if any(w in q_lower for w in ["grade", "evaluate", "my answer", "score my", "check my answer"]):
            return {"action": "evaluate_answer", "search_query": user_query}

        # Structured routing query to Ollama
        prompt = f"Categorize this user request:\n\"{user_query}\"\nOutput valid JSON."
        try:
            raw = self.client.generate(prompt=prompt, system=ROUTER_SYSTEM_PROMPT, temperature=0.0, format_json=True)
            # Remove any stray code fences if present
            clean_raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            decision = json.loads(clean_raw)
            action = decision.get("action", "ask_question")
            search_q = decision.get("search_query", user_query)
            if action not in ["ask_question", "summarize", "generate_quiz", "evaluate_answer", "general_chat"]:
                action = "ask_question"
            return {"action": action, "search_query": search_q or user_query}
        except Exception:
            # Fallback to standard QA
            return {"action": "ask_question", "search_query": user_query}

    def run(
        self,
        user_query: str,
        document_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Main execution loop.

        Args:
            user_query: The student's prompt or question.
            document_chunks: Extracted chunks across all uploaded documents.
            conversation_history: List of previous turns [{"role": "user"|"assistant", "content": "..."}].

        Returns:
            dict: {
                "answer": str,
                "trace": List[Dict[str, Any]],
                "sources": List[Dict[str, Any]],
                "action_taken": str
            }
        """
        self.trace = []
        step = 1

        if not user_query.strip():
            return {
                "answer": "Please enter a question or study command.",
                "trace": [],
                "sources": [],
                "action_taken": "none"
            }

        # Step 1: Route Intent
        routing = self.route_intent(user_query)
        action = routing["action"]
        search_q = routing.get("search_query", user_query)

        self._record_trace(
            step=step,
            tool="router",
            input_summary=f"Query: '{user_query[:60]}...'",
            result_summary=f"Selected action: '{action}' with search focus: '{search_q[:40]}'",
            status="success"
        )
        step += 1

        # Validation: Check if documents exist for doc-dependent actions
        if action in ["ask_question", "summarize", "generate_quiz"] and not document_chunks:
            return {
                "answer": "No documents are currently uploaded. Please upload a PDF, TXT, or Markdown lecture note first.",
                "trace": self.trace,
                "sources": [],
                "action_taken": action
            }

        # -------------------------------------------------------------
        # Action Branch 1: Document Question Answering (RAG)
        # -------------------------------------------------------------
        if action == "ask_question":
            retrieved = search_documents(query=search_q, chunks=document_chunks, top_k=3)

            if not retrieved:
                self._record_trace(
                    step=step,
                    tool="search_documents",
                    input_summary=f"Keyword search for: '{search_q}'",
                    result_summary="0 matching chunks found.",
                    status="failure"
                )
                return {
                    "answer": "I searched your uploaded documents, but could not find any sections matching your query terms.",
                    "trace": self.trace,
                    "sources": [],
                    "action_taken": action
                }

            source_tags = [f"{r['filename']} (p. {r['page_num']})" for r in retrieved]
            self._record_trace(
                step=step,
                tool="search_documents",
                input_summary=f"Query: '{search_q}'",
                result_summary=f"Retrieved {len(retrieved)} passages: {', '.join(source_tags)}",
                status="success"
            )
            step += 1

            prompt = format_qa_prompt(query=user_query, retrieved_chunks=retrieved)
            try:
                answer = self.client.generate(prompt=prompt, system=QA_SYSTEM_PROMPT, temperature=0.2)
                self._record_trace(
                    step=step,
                    tool="ollama_generate",
                    input_summary=f"Prompt context ({len(prompt.split())} words)",
                    result_summary=f"Synthesized response ({len(answer.split())} words)",
                    status="success"
                )
            except Exception as e:
                answer = f"Error generating answer: {str(e)}"
                self._record_trace(step=step, tool="ollama_generate", input_summary="LLM Synthesis", result_summary=str(e), status="failure")

            return {
                "answer": answer,
                "trace": self.trace,
                "sources": retrieved,
                "action_taken": action
            }

        # -------------------------------------------------------------
        # Action Branch 2: Document Summarization
        # -------------------------------------------------------------
        elif action == "summarize":
            # For summarization, use all chunks up to 1500 words or top-matched chunks
            relevant = search_documents(query=search_q, chunks=document_chunks, top_k=5)
            passages_to_use = relevant if relevant else document_chunks[:5]

            self._record_trace(
                step=step,
                tool="summarize_document",
                input_summary=f"Summarizing {len(passages_to_use)} passages",
                result_summary="Dispatched context to summarizer tool",
                status="success"
            )
            step += 1

            res = summarize_document(text_or_chunks=passages_to_use, client=self.client)
            if res["error"]:
                self._record_trace(step=step, tool="summarize_document", input_summary="Summarize", result_summary=res["error"], status="failure")
                return {"answer": f"Summarization failed: {res['error']}", "trace": self.trace, "sources": passages_to_use, "action_taken": action}

            return {"answer": res["summary"], "trace": self.trace, "sources": passages_to_use, "action_taken": action}

        # -------------------------------------------------------------
        # Action Branch 3: Practice Quiz Generation
        # -------------------------------------------------------------
        elif action == "generate_quiz":
            relevant = search_documents(query=search_q, chunks=document_chunks, top_k=4)
            passages_to_use = relevant if relevant else document_chunks[:4]

            self._record_trace(
                step=step,
                tool="generate_quiz",
                input_summary=f"Generating MCQs from {len(passages_to_use)} chunks",
                result_summary="Requesting structured MCQ schema",
                status="success"
            )
            step += 1

            res = generate_quiz(chunks=passages_to_use, client=self.client, num_questions=3)
            if res["error"]:
                self._record_trace(step=step, tool="generate_quiz", input_summary="Quiz Generator", result_summary=res["error"], status="failure")
                return {"answer": f"Could not generate quiz: {res['error']}", "trace": self.trace, "sources": passages_to_use, "action_taken": action}

            # Format the output into clean study markdown
            formatted_output = ["### Practice Quiz\n"]
            for q in res["questions"]:
                formatted_output.append(f"**Q{q['question_id']}: {q['question']}**")
                for opt in q["options"]:
                    formatted_output.append(f"- {opt}")
                formatted_output.append(f"\n*Correct Answer: **{q['correct_answer']}***")
                formatted_output.append(f"*Explanation:* {q['explanation']}\n")

            return {
                "answer": "\n".join(formatted_output),
                "trace": self.trace,
                "sources": passages_to_use,
                "action_taken": action,
                "quiz_raw": res["questions"]
            }

        # -------------------------------------------------------------
        # Action Branch 4: Answer Evaluation
        # -------------------------------------------------------------
        elif action == "evaluate_answer":
            # Attempt to retrieve reference context for the query
            ref_chunks = search_documents(query=user_query, chunks=document_chunks, top_k=3) if document_chunks else []

            self._record_trace(
                step=step,
                tool="evaluate_answer",
                input_summary=f"Evaluating student response with {len(ref_chunks)} reference chunks",
                result_summary="Grading against reference notes",
                status="success"
            )
            step += 1

            res = evaluate_answer(
                question="University Subject Question / Student Submission",
                student_answer=user_query,
                reference_chunks=ref_chunks,
                client=self.client
            )

            if res["error"]:
                return {"answer": f"Evaluation error: {res['error']}", "trace": self.trace, "sources": ref_chunks, "action_taken": action}

            formatted_feedback = f"""### Answer Evaluation
- **Estimated Score:** {res['score']} *({res['score_note']})*
- **Strengths:** {res['strengths']}
- **Areas for Improvement:** {res['gaps']}

**Sample Model Answer:**
> {res['model_answer']}
"""
            return {"answer": formatted_feedback, "trace": self.trace, "sources": ref_chunks, "action_taken": action}

        # -------------------------------------------------------------
        # Action Branch 5: General Conversational Assistant
        # -------------------------------------------------------------
        else:
            answer = self.client.generate(
                prompt=f"The student says: '{user_query}'. Respond politely and remind them how to upload documents or ask study questions.",
                system="You are UniStudy AI. Be helpful, concise, and academic."
            )
            return {"answer": answer, "trace": self.trace, "sources": [], "action_taken": "general_chat"}