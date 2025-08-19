import os
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Any
# OpenAI (raw SDK for embeddings + streaming chat)
from openai import OpenAI as OpenAIClient
# LangChain OpenAI (for LC chat model + embeddings)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Pinecone low-level + LC vectorstore wrapper
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
# Self-Query retriever bits
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.pinecone import PineconeTranslator


# ---------- Env & Clients ----------

load_dotenv()
OPEN_AI_API = os.getenv("OPEN_AI_API")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Raw OpenAI client (embeddings + streaming chat)
oa = OpenAIClient(api_key=OPEN_AI_API)

# Pinecone low-level client + index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


# ---------- App Container ----------

class App:
    pass

app = App()


# ---------- Init Services ----------

def init_services(app):
    # Embeddings for both LC vectorstore & manual dense calls
    app.embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPEN_AI_API,
    )

    # LC VectorStore over the same Pinecone index
    app.vectorstore = PineconeVectorStore(
        embedding=app.embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
        text_key="answer"  # <- your content lives in `metadata.answer`; VectorStore will read from this key
    )

    # LLM for SelfQueryRetriever (LangChain)
    app.lc_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPEN_AI_API
    )

    # --- Metadata schema & Self-Query configuration ---
    # Make the description sharply aligned with your corpus.
    # Your documents are short veterinary Q&A answers with:
    # - `answer` (free text guidance),
    # - `diseases` (one of a fixed set),
    # - `keywords` (comma-separated important terms).
    metadata_field_info = [
        AttributeInfo(
            name="diseases",
            type="string",
            description=(
                "Primary condition category for the answer. One of: "
                "Anxiety, Arthritis, Cancer, Chronic Disc Disease, Collapsed Trachea, Cushing's, "
                "Degenerative Myelopathy, Diet and Food, Ear Infections, Gastric Disorders, "
                "Kidney Disease, Liver Disease, Neurological, Heart Disease, Pancreatitis, "
                "Skin Disorders, Ticks Fleas Heartworm, Vaccinations"
            )
        ),
        AttributeInfo(
            name="keywords",
            type="string",
            description=(
                "Comma-separated key terms present in the Q&A (e.g., 'melanoma, biopsy, immunotherapy, turkey tail'). "
                "Use for matching specific treatments, supplements, or clinical concepts."
            )
        ),
    ]

    # Be very explicit about what the document text represents so the self-query
    # model writes precise filters and the right query text.
    document_content_description = (
        "Veterinary Q&A answer text for dogs/cats describing condition context, "
        "recommended care, conventional and holistic options, cautions, and follow-up steps. "
        "Each record includes: `answer` (free text), `diseases` (single category), "
        "`keywords` (comma-separated terms)."
    )

    # Translate the structured query to Pinecone filters
    translator = PineconeTranslator()

    app.self_query_retriever = SelfQueryRetriever.from_llm(
        llm=app.lc_llm,
        vectorstore=app.vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        structured_query_translator=translator,
        enable_limit=True,
        verbose=False,
    )


# ---------- Dense Retrieval (manual) ----------

def dense_retrieve(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top_k most relevant documents from Pinecone using manual dense embeddings.
    """
    # 1) Embed the query
    emb = oa.embeddings.create(
        model="text-embedding-3-large",
        input=query_text
    ).data[0].embedding

    # 2) Query Pinecone
    res = index.query(
        vector=emb,
        top_k=top_k,
        include_metadata=True
    )

    matches = getattr(res, "matches", None) or res.get("matches", [])
    out = []
    for m in matches:
        # m may be an object or dict depending on pinecone client version
        mid = getattr(m, "id", None) or m.get("id")
        score = getattr(m, "score", None) or m.get("score")
        md = getattr(m, "metadata", None) or m.get("metadata", {})

        out.append({
            "id": mid,
            "score": score,
            "answer": md.get("answer"),
            "diseases": md.get("diseases"),
            "keywords": md.get("keywords"),
            "source": "dense"
        })
    return out


# ---------- Self-Query Retrieval (LangChain) ----------

async def self_query_retrieve_async(query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Run the LangChain SelfQueryRetriever asynchronously and normalize docs.
    """
    # Make sure the retriever uses the caller's k
    app.self_query_retriever.search_kwargs["k"] = k

    # aget_relevant_documents returns List[Document]
    docs = await app.self_query_retriever.aget_relevant_documents(query_text)

    out = []
    for d in docs:
        md = d.metadata or {}
        out.append({
            "id": md.get("id") or md.get("doc_id") or md.get("uuid") or "",  # try common id fields; may be empty
            "score": md.get("score"),  # LC may not surface a score; keep if present
            "answer": md.get("answer") or d.page_content,  # fallback to content if needed
            "diseases": md.get("diseases"),
            "keywords": md.get("keywords"),
            "source": "self_query"
        })
    return out


# ---------- Parallel Orchestration ----------

async def retrieve_parallel(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Run dense (manual) and self-query (LC) in parallel, then merge and dedupe.
    """
    # dense_retrieve is sync & blocking -> run in a thread
    dense_task = asyncio.to_thread(dense_retrieve, query_text, top_k)
    sq_task = self_query_retrieve_async(query_text, top_k)

    dense_results, sq_results = await asyncio.gather(dense_task, sq_task)

    # Merge + dedupe by (id, answer) fallback (sometimes id may be missing)
    merged: Dict[str, Dict[str, Any]] = {}
    def key_of(doc):
        # Prefer stable id if present, else hash on answer text
        return doc.get("id") or f"ans::{(doc.get('answer') or '')[:64]}"

    for lst in (dense_results, sq_results):
        for d in lst:
            k = key_of(d)
            if k not in merged:
                merged[k] = d
            else:
                # Keep the best score if available, and merge sources
                old = merged[k]
                # pick higher score (if both present)
                if (d.get("score") or -1) > (old.get("score") or -1):
                    merged[k] = d
                else:
                    # append source tag
                    old_src = set((old.get("source") or "").split("+")) if old.get("source") else set()
                    new_src = set((d.get("source") or "").split("+")) if d.get("source") else set()
                    old["source"] = "+".join(sorted(old_src.union(new_src))) if old_src or new_src else None

    # Sort: score desc, then self_query first (often more precise filtering), then dense
    def sort_key(doc):
        src = doc.get("source") or ""
        src_rank = 0 if "self_query" in src else 1
        return (-1 * (doc.get("score") or 0), src_rank)

    results = sorted(merged.values(), key=sort_key)
    # Trim to a sensible combined top_k * 2 (you can change this)
    return results[: max(top_k, 10)]


# ---------- Streaming LLM ----------

def stream_llm_response(query_text: str, docs: List[Dict[str, Any]]):
    """
    Stream OpenAI GPT response using retrieved docs as context.
    """
    context_chunks = []
    for i, doc in enumerate(docs, 1):
        context_chunks.append(
            f"[{i}] Source: {doc.get('source','?')} | Score: {doc.get('score')}\n"
            f"Diseases: {doc.get('diseases')}\n"
            f"Keywords: {doc.get('keywords')}\n"
            f"Answer:\n{doc.get('answer')}\n"
        )
    context_text = "\n".join(context_chunks)

    import textwrap

    system_prompt = textwrap.dedent("""
OVERVIEW AND ROLE
Act as Dr. Steve DVM, advising pet parents. Synthesize Dr. Steve’s recommendations using answers from similar cases and uploaded content (protocols, videos, etc) to address inquiries.

Use each question to identify conditions, topics, symptom pattern, or cause — match it to similar cases/topics. (Ignore phrasing differences). Identify patterns (symptoms, energetics, diet, treatment stage), match them to similar cases, then synthesize a complete step-by-step plan in Dr. Steve’s style. Mirror Dr. Steve’s stepwise logic and prioritize diet, energetics, symptom progression, and formula-first sequencing. Do not copy an old answer, but compile all his answers from similar cases and synthesize an answer with the full reasoning behind Dr. Steve’s decisions — expanding each recommendation to the fullest explanation if he had more time. You may respond to any part of a question that matches past cases, and synthesize protocols across cases when relevant — as long as each step follows how Dr. Steve built his treatment path.

CLARIFY SYMPTOMS AND TREATMENT HISTORY
    • Confirm Condition: Ask only if you need to figure out what the main problem is. If you already know the main problem, start helping with that — even if other parts are unclear. If test results and symptoms don’t match, go with the steps Dr. Steve gives for cases with those symptoms — even if the test is normal and explain why the mismatch.
    • Confirm step in the decision tree: When prior treatments are unclear, ask targeted history questions to locate the next step in Dr. Steve’s sequence, otherwise start at the first step Dr. Steve normally recommends in similar cases/topics.
    • Never ask generic or unrelated questions.

Examples:
• Condition → "Is the skin red and inflamed or dry and pale?"
• Stage → "Have you tried Four Marvels yet?"

NEXT STEPS & SCOPE
Default to offer only 3–4 next steps and any therapies (acupuncture, chiropractic, laser, etc) he normally suggests, unless Dr. Steve’s original answer gave a full protocol/plan. Always include relevant tips. Also add the next step if it is a Gold Standard Herbs product. However, include additional steps to address each separate concern the user mentioned — or would influence the outcome — you may include it, but only if Dr. Steve has done so in similar cases. Match the user’s current place in the treatment plan and build forward from there. Do not re-recommend formulas, unless he has done so in a similar situation and if so explain the logic (dosing, combining, etc). Avoid showing future options or fallback plans — unless Dr. Steve modeled this. However, you may end with:

"Try that and let us know how it goes — there are other options I can recommend."

PROTOCOL SEQUENCE & CONTENT
Use the same sequencing Dr. Steve follows in similar cases:
Diet → Primary formulas → Adjustment formulas → Add-on or supports (if Dr. Steve refers to it)
Do not skip or rearrange these steps. If a user starts partway through this sequence, you must first clarify what they’ve already done. Clarify what the user has already done if the next step depends on it. If not, begin with the first step Dr. Steve normally uses in similar cases.

Every treatment recommendation must be explained with the same detailed reasoning Dr. Steve uses: cause-and-effect relationship, scientific, and on how each step contributes to resolution. If he used a step to confirm a condition, you must reflect that logic. If some parts are unclear, give the steps Dr. Steve normally uses for the parts you do know. Always start with the main condition first, then mention if you need more info for the other parts.

Always recommend each of these elements, if Dr. Steve does in similar cases:
• All relevant real food dietary changes/plan (including: 1/6 plant material, composition, specific probiotics and/or fermented vegetables, etc.)
• All herbal formulas (Gold Standard Herbs, Kan Essentials, Natural Path, etc)
• All supplements, supportive add-ons, and/or therapies (acupuncture, chiropractic, laser, etc)
- Any nuances, tips, and any warnings or exceptions Dr. Steve gave about certain treatments

Do not introduce any formula, brand, or approach unless it was clearly recommended by Dr. Steve in a similar case with matching logic. 

When recommending or reinforcing a formula, always include:
• The exact name of the formula
• The brand (Gold Standard Herbs, Kan Essentials, Natural Path, etc)
• Hyperlink each product, dosing chart, or supplement if Dr. Steve included one.
– For Gold Standard Herbs, always only use "https://goldstandardherbs.com/products", not links in uploaded content.
– For Kan Essentials, direct users to "Email Aleksandra Topic at aleks.topic.1@gmail.com".

BUILD FROM CURRENT CONTEXT
If a user’s question relates to a condition, symptom pattern, or topic addressed by Dr. Steve — proceed to answer using his established logic. Focus on recognizing the conditions and building a response from similar cases/topics/conditions.

Start from what’s already been done: diet, herbs, or supplements. Never re-recommend something unless Dr. Steve has done so in similar cases. If the user asks whether to continue a formula, you must respond using similar case patterns. If Dr. Steve has used the withdrawal or continuation of a formula as a diagnostic clue, include that logic if applicable.

If a formula was used but didn’t work, and Dr. Steve has questioned dosing for a similar situation, you may address it as Dr. Steve has done and include any relevant links.

FINAL/QUALITY CHECK & LIMITS
Cross-check every recommendation against Dr. Steve’s documented answers—make sure it’s 100% accurate, matches his logic step-for-step, and leaves no relevant information out.
""").strip()


    user_prompt = (
        f"User Question:\n{query_text}\n\n"
        f"Relevant Context (top results from two retrievers):\n{context_text}\n\n"
        f"Final Answer (cite context by bracket numbers like [1], [2] where relevant):"
    )

    stream = oa.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n--- Stream Completed ---")


# ---------- Main ----------

if __name__ == "__main__":
    init_services(app)

    query = (
        "Hi Dr. Steve... I just found out today my oldest Norfolk has a small mass "
        "taken for biopsy during a dental. Came back as oral melanoma (<4 mitoses). "
        "Vet ordered immunohistochemistry. I'm leaning holistic—using turkey tail (Mycodog) "
        "but stool is softer. What would you recommend while we wait 10–14 days?"
    )

    # Run the two retrievers in parallel and merge results
    results = asyncio.run(retrieve_parallel(query, top_k=5))

    print("\n--- Combined Retrieved Context (Self-Query + Dense) ---\n")
    for r in results:
        print(f"ID: {r.get('id')} | Score: {r.get('score')} | Source: {r.get('source')}")
        print(f"Diseases: {r.get('diseases')}")
        print(f"Keywords: {r.get('keywords')}")
        print(f"Answer: {r.get('answer')[:300]}...\n")
        print("-" * 80)

    print("\n--- Streaming LLM Response ---\n")
    stream_llm_response(query, results)

