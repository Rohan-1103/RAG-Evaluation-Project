"""
src/rag/pipeline.py

RAGPipeline — orchestrates retrieval and generation for one question.

Flow per question:
    RAGRequest
        ↓ EmbeddingManager.embed_query()   (query → vector)
        ↓ BaseVectorStore.query()           (vector → RetrievedChunks)
        ↓ _build_prompt()                   (chunks + question → prompt)
        ↓ Gemini.generate_content()         (prompt → answer)
        ↓ RAGResponse

Flow for a full dataset:
    EvalDataset (pairs with status=PENDING)
        ↓ answer_dataset()
        ↓ For each QAPair:
            ↓ answer_pair()
            ↓ pair.set_answer()       (PENDING → ANSWERED)
        ↓ RAGBatchResult

Design principles:
  - The pipeline is a pure orchestrator. Embedding logic lives in
    EmbeddingManager. Retrieval logic lives in BaseVectorStore.
    Generation logic lives in the Gemini SDK call.
    The pipeline wires them together — nothing more.
  - All collaborators are injected. The pipeline never constructs
    ChromaVectorStore or EmbeddingManager internally.
  - answer_pair() and answer_dataset() are the two public interfaces.
    The ComparisonRunner calls answer_dataset() with a ModelRunConfig.
    The FastAPI route calls answer_pair() for single-question UI.
  - Every call logs latency split into retrieval_latency_ms and
    generation_latency_ms separately. The dashboard shows both —
    high retrieval latency points to embedding model issues,
    high generation latency points to LLM issues.
  - Empty context is not an error — it is a data quality signal.
    The pipeline generates an answer from empty context and sets
    RAGResponse.empty_context=True. The evaluator will score
    faithfulness low — which is the correct behaviour.

Prompt design:
  The RAG generation prompt has three parts:
    1. System instruction — role, constraints, citation format
    2. Context block     — numbered retrieved chunks with citations
    3. Question          — the actual question to answer

  The system instruction explicitly tells the model:
    - Answer only from the provided context
    - If context is insufficient, say so explicitly
    - Cite sources by filename and page number
  This reduces hallucination and improves faithfulness scores.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import EmbeddingConfig, Settings
from src.dataset.schema import EvalDataset, QAPair, QAPairStatus
from src.rag.schema import (
    ModelRunConfig,
    RAGBatchResult,
    RAGPromptComponents,
    RAGRequest,
    RAGResponse,
    RetrievedChunk,
)
from src.vectorstore.base import BaseVectorStore, QueryResult
from src.vectorstore.embeddings import EmbeddingManager

# ===========================================================================
# PROMPT TEMPLATES
# ===========================================================================

_SYSTEM_INSTRUCTION = """You are a precise question-answering assistant.
Your task is to answer the question using ONLY the information provided \
in the context chunks below.

STRICT RULES:
1. Base your answer exclusively on the provided context.
2. If the context does not contain enough information to answer \
the question, respond with:
   "I cannot answer this question based on the provided context."
3. Do NOT use prior knowledge or make assumptions beyond the context.
4. Be concise and direct — 1 to 4 sentences unless more detail is \
explicitly needed.
5. When citing information, reference the source naturally \
(e.g. "According to [filename]...").
6. Do not repeat the question in your answer."""

_CONTEXT_BLOCK_TEMPLATE = """CONTEXT:
{context}

---"""

_QUESTION_TEMPLATE = """QUESTION:
{question}

ANSWER:"""

# ===========================================================================
# RAG PIPELINE
# ===========================================================================

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Retrieves relevant context from a vector store and generates
    an answer using a Gemini model for a given question.

    Constructor injection — all collaborators are passed in.
    Use RAGPipeline.from_settings() for the standard construction.

    Usage (single question):
        pipeline = RAGPipeline.from_settings(settings)
        response = pipeline.answer(
            RAGRequest(
                question="What is the leave policy?",
                collection_name="hr_docs",
                top_k=5,
            )
        )

    Usage (full dataset):
        batch_result = pipeline.answer_dataset(
            dataset=pending_dataset,
            run_config=ModelRunConfig(
                model_id="gemini-1.5-flash",
                collection_name="hr_docs",
                top_k=5,
            ),
        )
    """

    # Characters per token — used for prompt size estimation
    _CHARS_PER_TOKEN: float = 4.0

    # Maximum characters in the context block before truncation
    # Prevents exceeding model context limits on very high top_k
    _MAX_CONTEXT_CHARS: int = 24_000

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: BaseVectorStore,
        settings: Settings,
    ) -> None:
        """
        Initialise the RAG pipeline.

        Args:
            embedding_manager: For embedding queries at retrieval time.
            vector_store:      For similarity search.
            settings:          Application settings (model, API key, etc.)
        """
        self._embedding_manager = embedding_manager
        self._vector_store = vector_store
        self._settings = settings
        self._model_cache: dict[str, Any] = {}
        # model_id → google.generativeai.GenerativeModel

        self._configure_gemini()

        logger.info(
            f"RAGPipeline initialised. "
            f"embedding_model={embedding_manager.model_name}, "
            f"embedding_dim={embedding_manager.dimension}"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        vector_store: BaseVectorStore | None = None,
    ) -> RAGPipeline:
        """
        Standard factory — build RAGPipeline from application Settings.

        Args:
            settings:     Application Settings from get_settings().
            vector_store: Optional pre-built vector store.
                          If None, builds ChromaVectorStore from settings.

        Returns:
            Fully wired RAGPipeline instance.
        """
        embedding_manager = EmbeddingManager(settings.embedding)

        if vector_store is None:
            from src.vectorstore.chroma import ChromaVectorStore
            vector_store = ChromaVectorStore(
                config=settings.chroma,
                embedding_manager=embedding_manager,
            )

        return cls(
            embedding_manager=embedding_manager,
            vector_store=vector_store,
            settings=settings,
        )

    # ------------------------------------------------------------------
    # Gemini initialisation
    # ------------------------------------------------------------------

    def _configure_gemini(self) -> None:
        """Configure Gemini API key once at pipeline construction."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._settings.gemini.api_key)
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is not installed. "
                "Run: poetry add google-generativeai"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"RAGPipeline: Failed to configure Gemini API: {exc}"
            ) from exc

    def _get_model(
        self,
        model_id: str,
        temperature: float,
        max_output_tokens: int,
    ) -> Any:
        """
        Get or create a Gemini GenerativeModel for a given config.

        Models are cached by (model_id, temperature, max_output_tokens)
        so the same configuration reuses the same model instance
        across all pairs in a batch run.
        """
        cache_key = f"{model_id}|{temperature}|{max_output_tokens}"

        if cache_key not in self._model_cache:
            try:
                import google.generativeai as genai

                generation_config = genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )

                self._model_cache[cache_key] = genai.GenerativeModel(
                    model_name=model_id,
                    generation_config=generation_config,
                    system_instruction=_SYSTEM_INSTRUCTION,
                )

                logger.debug(
                    f"RAGPipeline: Created model instance "
                    f"'{model_id}' "
                    f"(temperature={temperature}, "
                    f"max_tokens={max_output_tokens})"
                )

            except Exception as exc:
                raise RAGPipelineError(
                    reason=(
                        f"Failed to create Gemini model '{model_id}': "
                        f"{exc}"
                    ),
                    original_exception=exc,
                ) from exc

        return self._model_cache[cache_key]

    # ------------------------------------------------------------------
    # Public interfaces
    # ------------------------------------------------------------------

    def answer(
        self,
        request: RAGRequest,
        model_id: str | None = None,
        log_prompt: bool = False,
    ) -> RAGResponse:
        """
        Answer a single question via RAG.

        Args:
            request:    RAGRequest with question and retrieval params.
            model_id:   Override model ID. Defaults to
                        settings.dataset_gen.model
                        (use a dedicated RAG model setting in production).
            log_prompt: If True, store the full prompt in RAGResponse
                        for debugging. Avoid in production — prompts
                        can be large.

        Returns:
            RAGResponse with generated answer and retrieved chunks.
            Never raises — errors are captured in the response.
        """
        resolved_model_id = (
            model_id or self._settings.dataset_gen.model
        )
        wall_start = time.monotonic()

        try:
            # Stage 1: Embed query
            embed_start = time.monotonic()
            query_vector = self._embed_query(request.question)
            embed_latency_ms = (
                time.monotonic() - embed_start
            ) * 1000

            # Stage 2: Retrieve chunks
            retrieve_start = time.monotonic()
            retrieved_chunks = self._retrieve(
                query_vector=query_vector,
                collection_name=request.collection_name,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
            )
            retrieval_latency_ms = (
                time.monotonic() - retrieve_start
            ) * 1000 + embed_latency_ms

            empty_context = len(retrieved_chunks) == 0

            if empty_context:
                logger.warning(
                    f"RAGPipeline: No chunks retrieved for question "
                    f"'{request.question[:60]}...' "
                    f"from collection '{request.collection_name}'. "
                    f"score_threshold={request.score_threshold}. "
                    f"Answer will be generated without context."
                )

            # Stage 3: Build prompt
            prompt_components = self._build_prompt(
                question=request.question,
                retrieved_chunks=retrieved_chunks,
            )

            # Stage 4: Generate answer
            gen_start = time.monotonic()
            raw_answer, input_tokens, output_tokens, refused = (
                self._generate(
                    prompt=prompt_components.full_prompt,
                    model_id=resolved_model_id,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                )
            )
            generation_latency_ms = (
                time.monotonic() - gen_start
            ) * 1000

            total_latency_ms = (
                time.monotonic() - wall_start
            ) * 1000

            # Stage 5: Estimate cost
            estimated_cost = self._estimate_cost(
                model_id=resolved_model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            logger.debug(
                f"RAGPipeline.answer: "
                f"question='{request.question[:40]}...', "
                f"model='{resolved_model_id}', "
                f"n_retrieved={len(retrieved_chunks)}, "
                f"retrieval={retrieval_latency_ms:.0f}ms, "
                f"generation={generation_latency_ms:.0f}ms, "
                f"tokens={input_tokens}+{output_tokens}"
            )

            return RAGResponse(
                request=request,
                rag_model=resolved_model_id,
                answered_at=datetime.now(timezone.utc),
                generated_answer=raw_answer,
                retrieved_chunks=retrieved_chunks,
                retrieval_latency_ms=round(retrieval_latency_ms, 1),
                generation_latency_ms=round(generation_latency_ms, 1),
                total_latency_ms=round(total_latency_ms, 1),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimated_cost,
                empty_context=empty_context,
                answer_refused=refused,
                prompt_used=(
                    prompt_components.full_prompt if log_prompt else None
                ),
            )

        except Exception as exc:
            total_latency_ms = (
                time.monotonic() - wall_start
            ) * 1000
            logger.error(
                f"RAGPipeline.answer: Unexpected error for question "
                f"'{request.question[:60]}': {exc}"
            )
            # Return a failed response — never raise to caller
            return self._make_error_response(
                request=request,
                model_id=resolved_model_id,
                total_latency_ms=total_latency_ms,
                error=str(exc),
            )

    def answer_pair(
        self,
        pair: QAPair,
        run_config: ModelRunConfig,
    ) -> RAGResponse:
        """
        Answer a single QAPair and update its status to ANSWERED.

        Constructs a RAGRequest from the pair and run_config,
        calls answer(), and calls pair.set_answer() to transition
        the pair from PENDING → ANSWERED.

        Args:
            pair:       QAPair with status=PENDING.
            run_config: ModelRunConfig specifying retrieval parameters.

        Returns:
            RAGResponse for traceability and logging.

        Raises:
            ValueError: if pair.status is not PENDING.
        """
        if pair.status != QAPairStatus.PENDING:
            raise ValueError(
                f"RAGPipeline.answer_pair: QAPair '{pair.id}' "
                f"has status='{pair.status.value}'. "
                f"Only PENDING pairs can be answered."
            )

        request = run_config.to_rag_request(
            question=pair.question,
            pair_id=pair.id,
        )

        response = self.answer(
            request=request,
            model_id=run_config.model_id,
        )

        # Transition pair PENDING → ANSWERED
        pair.set_answer(
            generated_answer=response.generated_answer,
            retrieved_chunks=response.chunk_contents,
            retrieved_chunk_sources=response.chunk_sources,
            latency_ms=response.total_latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            rag_model=run_config.model_id,
        )

        return response

    def answer_dataset(
        self,
        dataset: EvalDataset,
        run_config: ModelRunConfig,
        on_pair_complete: Any | None = None,
    ) -> RAGBatchResult:
        """
        Run all PENDING pairs in a dataset through the RAG pipeline.

        Processes pairs sequentially. Each answered pair is transitioned
        PENDING → ANSWERED in place on the dataset object.

        Args:
            dataset:          EvalDataset with PENDING pairs.
            run_config:       ModelRunConfig with retrieval parameters.
            on_pair_complete: Optional callable(pair_idx, total, response)
                              called after each pair for progress tracking.

        Returns:
            RAGBatchResult with aggregate statistics.
            The dataset object is mutated in place — pairs are now ANSWERED.
        """
        pending_pairs = [
            p for p in dataset.pairs
            if p.status == QAPairStatus.PENDING
        ]

        if not pending_pairs:
            logger.warning(
                f"RAGPipeline.answer_dataset: No PENDING pairs in "
                f"dataset '{dataset.name}'. "
                f"All pairs may already be answered."
            )
            return RAGBatchResult(
                rag_model=run_config.model_id,
                collection_name=run_config.collection_name,
                top_k=run_config.top_k,
                temperature=run_config.temperature,
                total_pairs=len(dataset.pairs),
                answered_pairs=0,
                failed_pairs=0,
                skipped_pairs=len(dataset.pairs),
                total_latency_ms=0.0,
                avg_latency_ms=0.0,
            )

        total = len(pending_pairs)
        logger.info(
            f"RAGPipeline.answer_dataset: Processing {total} pairs. "
            f"model='{run_config.model_id}', "
            f"collection='{run_config.collection_name}', "
            f"top_k={run_config.top_k}"
        )

        wall_start = time.monotonic()
        responses: list[RAGResponse] = []
        answered_count = 0
        failed_count = 0
        empty_context_count = 0
        refused_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        all_similarity_scores: list[float] = []

        for pair_idx, pair in enumerate(pending_pairs):
            try:
                response = self.answer_pair(
                    pair=pair,
                    run_config=run_config,
                )
                responses.append(response)
                answered_count += 1

                if response.empty_context:
                    empty_context_count += 1
                if response.answer_refused:
                    refused_count += 1

                total_input_tokens += response.input_tokens
                total_output_tokens += response.output_tokens
                total_cost += response.estimated_cost_usd

                for chunk in response.retrieved_chunks:
                    all_similarity_scores.append(
                        chunk.similarity_score
                    )

                logger.debug(
                    f"RAGPipeline.answer_dataset: "
                    f"Pair {pair_idx + 1}/{total} answered. "
                    f"latency={response.total_latency_ms:.0f}ms"
                )

                if on_pair_complete is not None:
                    try:
                        on_pair_complete(
                            pair_idx + 1,
                            total,
                            response,
                        )
                    except Exception as cb_exc:
                        logger.warning(
                            f"RAGPipeline: on_pair_complete callback "
                            f"failed: {cb_exc}"
                        )

            except Exception as exc:
                failed_count += 1
                pair.mark_failed(
                    f"RAGPipeline error: {type(exc).__name__}: {exc}"
                )
                logger.error(
                    f"RAGPipeline.answer_dataset: Pair '{pair.id}' "
                    f"failed: {exc}"
                )

        total_latency_ms = (
            time.monotonic() - wall_start
        ) * 1000
        avg_latency_ms = (
            total_latency_ms / answered_count
            if answered_count > 0
            else 0.0
        )
        avg_similarity = (
            sum(all_similarity_scores) / len(all_similarity_scores)
            if all_similarity_scores
            else 0.0
        )

        batch_result = RAGBatchResult(
            rag_model=run_config.model_id,
            collection_name=run_config.collection_name,
            top_k=run_config.top_k,
            temperature=run_config.temperature,
            total_pairs=total,
            answered_pairs=answered_count,
            failed_pairs=failed_count,
            skipped_pairs=0,
            total_latency_ms=round(total_latency_ms, 1),
            avg_latency_ms=round(avg_latency_ms, 1),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_estimated_cost_usd=round(total_cost, 6),
            empty_context_count=empty_context_count,
            refused_count=refused_count,
            avg_similarity_score=round(avg_similarity, 4),
            responses=responses,
        )

        logger.info(
            f"RAGPipeline.answer_dataset: Complete. "
            f"{batch_result.summary()}"
        )

        return batch_result

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_query(self, question: str) -> Any:
        """
        Embed a query string into a vector.

        Uses EmbeddingManager.embed_query() which applies the
        query-specific task type for asymmetric search models.
        """
        try:
            return self._embedding_manager.embed_query(question)
        except Exception as exc:
            raise RAGPipelineError(
                reason=f"Query embedding failed: {exc}",
                original_exception=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        query_vector: Any,
        collection_name: str,
        top_k: int,
        score_threshold: float,
    ) -> list[RetrievedChunk]:
        """
        Query the vector store and return typed RetrievedChunks.

        Converts QueryResult objects (from BaseVectorStore) to
        RetrievedChunk objects (from RAG schema).

        Returns empty list if:
          - Collection does not exist (logged as error)
          - No chunks meet the score_threshold
          - Vector store returns an empty result set
        """
        try:
            if not self._vector_store.collection_exists(collection_name):
                logger.error(
                    f"RAGPipeline._retrieve: Collection "
                    f"'{collection_name}' does not exist. "
                    f"Run the ingestion pipeline first."
                )
                return []

            query_results: list[QueryResult] = (
                self._vector_store.query(
                    query_embedding=query_vector,
                    collection_name=collection_name,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
            )

            retrieved: list[RetrievedChunk] = []
            for qr in query_results:
                retrieved.append(
                    RetrievedChunk(
                        content=qr.document.page_content,
                        source_file=qr.document.metadata.source_file,
                        source_page=qr.document.metadata.page_number,
                        chunk_index=qr.document.chunk_index,
                        similarity_score=qr.similarity_score,
                        rank=qr.rank,
                        collection_name=collection_name,
                    )
                )

            return retrieved

        except Exception as exc:
            logger.error(
                f"RAGPipeline._retrieve: Vector store query failed "
                f"for collection '{collection_name}': {exc}"
            )
            return []

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> RAGPromptComponents:
        """
        Construct the complete RAG generation prompt.

        Structure:
            [system_instruction]   — role + constraints
            [context_block]        — numbered retrieved chunks
            [question]             — the actual question

        Context is formatted with source citations so the model
        can reference them in its answer and the evaluator can
        verify faithfulness.

        If retrieved_chunks is empty, the context block tells the
        model explicitly — this triggers the "I cannot answer"
        response rather than hallucination.
        """
        # Build context block
        if retrieved_chunks:
            context_entries = []
            total_chars = 0

            for chunk in retrieved_chunks:
                entry = f"{chunk.rank}. [{chunk.citation}]\n{chunk.content}"
                entry_chars = len(entry)

                # Truncate context if it would exceed max size
                if (
                    total_chars + entry_chars > self._MAX_CONTEXT_CHARS
                    and context_entries
                ):
                    logger.debug(
                        f"RAGPipeline._build_prompt: Context truncated "
                        f"at chunk {chunk.rank} to stay within "
                        f"{self._MAX_CONTEXT_CHARS} chars."
                    )
                    break

                context_entries.append(entry)
                total_chars += entry_chars

            context_text = "\n\n".join(context_entries)
        else:
            context_text = (
                "No relevant context was found in the knowledge base "
                "for this question."
            )

        context_block = _CONTEXT_BLOCK_TEMPLATE.format(
            context=context_text
        )
        question_block = _QUESTION_TEMPLATE.format(
            question=question
        )

        full_prompt = f"{context_block}\n\n{question_block}"

        return RAGPromptComponents(
            system_instruction=_SYSTEM_INSTRUCTION,
            context_block=context_block,
            question=question,
            full_prompt=full_prompt,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _call_gemini(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_output_tokens: int,
    ) -> tuple[str, int, int]:
        """
        Make a single Gemini generation API call with retry.

        Returns (answer_text, input_tokens, output_tokens).

        Decorated with @retry for 429 and 503 handling.
        reraise=True means the final failure propagates to _generate()
        which converts it to an error response.
        """
        model = self._get_model(
            model_id=model_id,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        response = model.generate_content(prompt)

        # Extract token counts
        input_tokens = 0
        output_tokens = 0
        if (
            hasattr(response, "usage_metadata")
            and response.usage_metadata
        ):
            input_tokens = getattr(
                response.usage_metadata,
                "prompt_token_count",
                0,
            ) or 0
            output_tokens = getattr(
                response.usage_metadata,
                "candidates_token_count",
                0,
            ) or 0

        # Extract text
        text = ""
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text = "".join(
                    part.text
                    for part in candidate.content.parts
                    if hasattr(part, "text")
                )

        return text, input_tokens, output_tokens

    def _generate(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_output_tokens: int,
    ) -> tuple[str, int, int, bool]:
        """
        Generate an answer from the prompt.

        Returns (answer_text, input_tokens, output_tokens, answer_refused).

        answer_refused=True if:
          - The model returned an empty response
          - The response contains a known refusal phrase
          - The API call failed after all retries

        Never raises — failures return an empty answer with refused=True.
        """
        try:
            text, input_tokens, output_tokens = self._call_gemini(
                prompt=prompt,
                model_id=model_id,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            logger.error(
                f"RAGPipeline._generate: Gemini call failed "
                f"after all retries: {exc}"
            )
            return "", 0, 0, True

        if not text or not text.strip():
            logger.warning(
                f"RAGPipeline._generate: Gemini returned empty response "
                f"for model '{model_id}'."
            )
            return "", input_tokens, output_tokens, True

        # Detect explicit refusals
        refused = self._is_refusal(text)
        if refused:
            logger.debug(
                f"RAGPipeline._generate: Model '{model_id}' refused "
                f"to answer. Response: '{text[:100]}...'"
            )

        return text.strip(), input_tokens, output_tokens, refused

    @staticmethod
    def _is_refusal(text: str) -> bool:
        """
        Detect if the model explicitly refused to answer.

        Checks for common refusal patterns. Not exhaustive —
        designed to catch explicit refusals, not low-quality answers.

        Refusal ≠ "I cannot answer based on context" (which is
        the correct behaviour for out-of-scope questions and should
        NOT be flagged as a refusal).
        """
        refusal_patterns = [
            "i cannot provide",
            "i'm unable to",
            "i am unable to",
            "i cannot assist with",
            "i can't help with",
            "this request violates",
            "i'm not able to",
            "i am not able to",
        ]

        text_lower = text.lower()
        return any(
            pattern in text_lower
            for pattern in refusal_patterns
        )

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def _estimate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate USD cost for one RAG call.

        Returns 0.0 if the model is not in the registry.
        """
        try:
            from config import get_model_registry

            registry = get_model_registry()
            model = registry.get_model(model_id)
            safety = registry.cost_estimation.safety_multiplier
            return model.estimate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                safety_multiplier=safety,
            )
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Error response construction
    # ------------------------------------------------------------------

    def _make_error_response(
        self,
        request: RAGRequest,
        model_id: str,
        total_latency_ms: float,
        error: str,
    ) -> RAGResponse:
        """
        Construct a RAGResponse for a completely failed pipeline call.

        Returns an empty-answer response with answer_refused=True
        and the error captured in the answer field for traceability.
        """
        return RAGResponse(
            request=request,
            rag_model=model_id,
            answered_at=datetime.now(timezone.utc),
            generated_answer=(
                f"[RAGPipeline Error: {error[:200]}]"
            ),
            retrieved_chunks=[],
            retrieval_latency_ms=0.0,
            generation_latency_ms=0.0,
            total_latency_ms=round(total_latency_ms, 1),
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            empty_context=True,
            answer_refused=True,
            prompt_used=None,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def warm_up(
        self,
        collection_name: str,
        model_id: str | None = None,
    ) -> bool:
        """
        Validate that the pipeline is fully operational.

        Runs a minimal end-to-end test:
          1. Embeds a probe query
          2. Queries the collection (top_k=1)
          3. Generates a one-token response

        Used by the FastAPI startup health check and the Streamlit
        sidebar "Test Pipeline" button.

        Returns:
            True if all stages passed, False otherwise.
        """
        resolved_model = model_id or self._settings.dataset_gen.model

        logger.info(
            f"RAGPipeline.warm_up: Testing pipeline. "
            f"collection='{collection_name}', "
            f"model='{resolved_model}'"
        )

        try:
            # Check 1: Embedding
            probe_vector = self._embed_query("pipeline warm-up probe")
            if probe_vector is None or len(probe_vector) == 0:
                logger.error("RAGPipeline.warm_up: Embedding failed.")
                return False

            # Check 2: Collection exists
            if not self._vector_store.collection_exists(collection_name):
                logger.error(
                    f"RAGPipeline.warm_up: Collection "
                    f"'{collection_name}' does not exist."
                )
                return False

            # Check 3: Retrieval (don't fail if no results — empty
            # collection is valid during initial testing)
            self._retrieve(
                query_vector=probe_vector,
                collection_name=collection_name,
                top_k=1,
                score_threshold=0.0,
            )

            # Check 4: Model is accessible (skip generation to
            # avoid token cost in warm-up)
            self._get_model(
                model_id=resolved_model,
                temperature=0.0,
                max_output_tokens=64,
            )

            logger.info("RAGPipeline.warm_up: All checks passed.")
            return True

        except Exception as exc:
            logger.error(
                f"RAGPipeline.warm_up: Failed with: {exc}"
            )
            return False

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_manager.dimension

    @property
    def embedding_model(self) -> str:
        return self._embedding_manager.model_name

    def __repr__(self) -> str:
        return (
            f"RAGPipeline("
            f"embedding='{self.embedding_model}', "
            f"dim={self.embedding_dimension}, "
            f"cached_models={list(self._model_cache.keys())})"
        )

# ===========================================================================
# CUSTOM EXCEPTION
# ===========================================================================

class RAGPipelineError(Exception):
    """
    Raised for unrecoverable RAG pipeline failures.

    Distinct from empty context (data quality issue) and
    generation refusal (model behaviour). RAGPipelineError means
    a component failed in a way that prevents any response:
      - Embedding model unreachable
      - Vector store connection failed
      - Gemini SDK not installed

    Caught by answer() and converted to an error RAGResponse.
    Never propagates to the caller.
    """

    def __init__(
        self,
        reason: str,
        original_exception: Exception | None = None,
    ) -> None:
        self.reason = reason
        self.original_exception = original_exception
        super().__init__(
            f"[RAGPipeline] {reason}"
            + (
                f" (caused by {type(original_exception).__name__}: "
                f"{original_exception})"
                if original_exception
                else ""
            )
        )
        
__all__ = [
    "RAGPipeline",
    "RAGPipelineError",
]