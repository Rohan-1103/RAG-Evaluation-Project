"""
scripts/verify_config.py

Smoke test for the entire config layer.
Run with: python scripts/verify_config.py

Expected output: all sections print without exceptions.
Exit code 0 = config layer is healthy.
Exit code 1 = something is broken — read the error.
"""

import sys
import traceback
from pathlib import Path

# Ensure project root is on sys.path regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force .env reload — overrides any stale Windows environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(label: str, value: object) -> None:
    print(f"  ✓  {label:<45} {value}")


def fail(label: str, exc: Exception) -> None:
    print(f"  ✗  {label}")
    print(f"     ERROR: {exc}")
    traceback.print_exc()
    sys.exit(1)


# ── 1. SETTINGS ──────────────────────────────────────────────────────────────
section("1 · Settings (config/settings.py)")
try:
    from config.settings import get_settings
    s = get_settings()
    check("app_env",                    s.app_env)
    check("is_development",             s.is_development)
    check("gemini.api_key (masked)",    f"{s.gemini.api_key[:6]}...{s.gemini.api_key[-4:]}")
    check("embedding.provider",         s.embedding.provider)
    check("embedding.model",            s.embedding.model)
    check("chroma.persist_dir",         s.chroma.persist_dir)
    check("storage.database_url",       s.storage.database_url)
    check("ingestion.chunk_size",       s.ingestion.chunk_size)
    check("ingestion.chunk_overlap",    s.ingestion.chunk_overlap)
    check("judge.model",                s.judge.model)
    check("judge.temperature",          s.judge.temperature)
    check("metric_weights.faithfulness",s.metric_weights.faithfulness)
    check("weights sum to 1.0",         round(
        s.metric_weights.faithfulness +
        s.metric_weights.answer_relevance +
        s.metric_weights.context_precision +
        s.metric_weights.correctness, 6
    ))
    check("dataset_gen.model",          s.dataset_gen.model)
    check("comparison.max_concurrent",  s.comparison.max_concurrent_runs)
    check("api.host:port",              f"{s.api.host}:{s.api.port}")
    check("phoenix.enabled",            s.phoenix.enabled)
    check("phoenix.collector_endpoint", s.phoenix.collector_endpoint)
    check("tracing_enabled",            s.tracing_enabled)
    print("\n  ✅  Settings: PASSED")
except Exception as exc:
    fail("Settings failed to load", exc)


# ── 2. DATA DIRECTORIES ───────────────────────────────────────────────────────
section("2 · Auto-created data directories")
try:
    from config.settings import get_settings
    s = get_settings()
    dirs = [
        s.chroma.persist_dir,
        s.storage.raw_docs_dir,
        s.storage.processed_docs_dir,
        s.storage.datasets_dir,
        s.storage.comparison_results_dir,
        s.logging.file.parent,
    ]
    for d in dirs:
        exists = d.exists()
        check(str(d), "EXISTS ✓" if exists else "MISSING ✗")
        if not exists:
            raise FileNotFoundError(
                f"Directory '{d}' was not created by Settings validator. "
                "Check create_required_directories() in settings.py."
            )
    print("\n  ✅  Directories: PASSED")
except Exception as exc:
    fail("Directory check failed", exc)


# ── 3. MODEL REGISTRY ─────────────────────────────────────────────────────────
section("3 · Model Registry (config/models.yaml)")
try:
    from config import get_model_registry
    reg = get_model_registry()

    check("providers loaded",           list(reg.providers.keys()))
    check("total models in registry",   len(reg.models))
    check("enabled models",             len(reg.get_enabled_models()))

    default_rag   = reg.get_default_rag_model()
    default_judge = reg.get_default_judge_model()
    default_gen   = reg.get_default_dataset_gen_model()
    default_cmp   = reg.get_default_comparison_models()

    check("default RAG model",          default_rag.id)
    check("default judge model",        default_judge.id)
    check("default dataset-gen model",  default_gen.id)
    check("default comparison models",  [m.id for m in default_cmp])

    # Test get_model raises KeyError on bad id
    try:
        reg.get_model("this-model-does-not-exist")
        fail("get_model should raise KeyError", Exception("Did not raise"))
    except KeyError:
        check("get_model raises KeyError on bad id", "OK")

    # Test role queries
    judge_models = reg.get_models_for_role("judge")
    rag_models   = reg.get_models_for_role("rag_pipeline")
    check("models recommended for 'judge'",         [m.id for m in judge_models])
    check("models recommended for 'rag_pipeline'",  [m.id for m in rag_models])

    # Test provider query
    google_models = reg.get_models_by_provider("google")
    check("google models (enabled)",    [m.id for m in google_models])

    # Test cost estimation
    cost_est = reg.estimate_comparison_cost(
        model_ids=[default_rag.id, default_judge.id],
        n_questions=10,
    )
    check("cost estimate (10 questions)", cost_est)

    # Test LLMModelConfig helpers
    flash = reg.get_model("gemini-1.5-flash")
    cost_single = flash.estimate_cost(input_tokens=1000, output_tokens=200)
    check("gemini-1.5-flash.estimate_cost(1k in, 200 out)", f"${cost_single:.6f}")
    check("gemini-1.5-flash.is_free_tier",   flash.is_free_tier)
    check("gemini-1.5-flash.context_window", flash.context_window)

    # Test UIConfig helpers
    color = reg.ui.color_for_provider("google")
    check("ui.color_for_provider('google')", color)
    band  = reg.ui.band_for_score(4.7)
    check("ui.band_for_score(4.7)",          band)

    print("\n  ✅  Model Registry: PASSED")
except Exception as exc:
    fail("Model Registry failed", exc)


# ── 4. EVAL CONFIG ────────────────────────────────────────────────────────────
section("4 · Eval Config (config/eval.yaml)")
try:
    from config import get_eval_config
    ec = get_eval_config()

    check("active prompt version",          ec.active_prompt_version)
    check("judge model (from config)",       ec.judge.model if hasattr(ec.judge, "model") else "N/A — from settings")
    check("judge score range",              f"{ec.judge.score_min} – {ec.judge.score_max}")
    check("judge max_response_tokens",      ec.judge.max_response_tokens)
    check("judge parse_failure_flag",       ec.judge.parse_failure_flag)

    # Metric weights
    weights = ec.metrics.weight_map
    check("metric weight map",              weights)
    check("weights sum to 1.0",             round(sum(weights.values()), 6))

    # Each metric
    for name, cfg in ec.metrics.as_dict().items():
        check(f"{name}.weight",             cfg.weight)
        check(f"{name}.rubric keys",        sorted(cfg.rubric.keys()))

    # Test get_metric raises KeyError on bad name
    try:
        ec.metrics.get_metric("nonexistent_metric")
        fail("get_metric should raise KeyError", Exception("Did not raise"))
    except KeyError:
        check("get_metric raises KeyError on bad name", "OK")

    # Test prompt rendering for each metric
    dummy_vars: dict[str, str] = {
        "question":         "What is RAG?",
        "answer":           "RAG stands for Retrieval Augmented Generation.",
        "context_chunks":   "1. RAG combines retrieval with generation.",
        "reference_answer": "RAG is a technique that retrieves documents.",
    }

    for name in ["faithfulness", "answer_relevance", "context_precision", "correctness"]:
        try:
            prompt = ec.get_metric_prompt(name, **dummy_vars)
            # Verify output_format_instruction was injected
            assert "JSON" in prompt, "output_format_instruction not injected"
            assert dummy_vars["question"] in prompt, "question not injected"
            check(f"{name} prompt renders OK (len)",  f"{len(prompt)} chars")
        except Exception as exc:
            fail(f"{name} prompt rendering failed", exc)

    # Eval run config
    check("eval_run.max_questions_per_run",         ec.eval_run.max_questions_per_run)
    check("eval_run.parallel_metrics",              ec.eval_run.parallel_metrics_per_question)
    check("eval_run.inter_question_delay",          ec.eval_run.inter_question_delay_seconds)
    check("eval_run.checkpoint_after_each",         ec.eval_run.checkpoint_after_each_question)

    # Composite score config
    check("composite_score.formula",                ec.composite_score.formula)
    check("composite_score.missing_strategy",       ec.composite_score.missing_metric_strategy)

    # Aggregation config
    check("aggregation.statistics",                 ec.aggregation.statistics)
    check("aggregation.min_q_reliable",             ec.aggregation.min_questions_for_reliable_aggregation)

    print("\n  ✅  Eval Config: PASSED")
except Exception as exc:
    fail("Eval Config failed", exc)


# ── 5. SINGLETON CACHE BEHAVIOUR ─────────────────────────────────────────────
section("5 · Singleton cache behaviour (lru_cache)")
try:
    from config import get_settings, get_model_registry, get_eval_config

    s1 = get_settings()
    s2 = get_settings()
    check("get_settings() returns same object", s1 is s2)

    r1 = get_model_registry()
    r2 = get_model_registry()
    check("get_model_registry() returns same object", r1 is r2)

    e1 = get_eval_config()
    e2 = get_eval_config()
    check("get_eval_config() returns same object", e1 is e2)

    # Verify cache_clear works (used in tests)
    get_settings.cache_clear()
    s3 = get_settings()
    check("cache_clear() forces re-construction", s3 is not s1)

    print("\n  ✅  Singleton cache: PASSED")
except Exception as exc:
    fail("Singleton cache test failed", exc)


# ── 6. CROSS-CONFIG CONSISTENCY ───────────────────────────────────────────────
section("6 · Cross-config consistency checks")
try:
    from config import get_settings, get_model_registry, get_eval_config
    # Re-fetch after cache_clear above
    get_settings.cache_clear()
    s   = get_settings()
    reg = get_model_registry()
    ec  = get_eval_config()

    # Judge model in settings must exist in registry
    judge_in_registry = reg._find_by_id(s.judge.model)
    check(
        f"settings.judge.model ('{s.judge.model}') exists in registry",
        "YES" if judge_in_registry else "NO — MISMATCH"
    )
    if judge_in_registry is None:
        raise ValueError(
            f"settings.judge.model='{s.judge.model}' is not in models.yaml. "
            "Either update JUDGE_MODEL in .env or add the model to models.yaml."
        )

    # Dataset gen model in settings must exist in registry
    gen_in_registry = reg._find_by_id(s.dataset_gen.model)
    check(
        f"settings.dataset_gen.model ('{s.dataset_gen.model}') exists in registry",
        "YES" if gen_in_registry else "NO — MISMATCH"
    )

    # Metric weights in settings must match eval.yaml
    settings_weights = {
        "faithfulness":     s.metric_weights.faithfulness,
        "answer_relevance": s.metric_weights.answer_relevance,
        "context_precision":s.metric_weights.context_precision,
        "correctness":      s.metric_weights.correctness,
    }
    eval_weights = ec.metrics.weight_map
    for metric, sw in settings_weights.items():
        ew = eval_weights[metric]
        match = abs(sw - ew) < 1e-6
        check(
            f"weight '{metric}' settings={sw} eval.yaml={ew}",
            "MATCH ✓" if match else f"MISMATCH ✗ — update .env or eval.yaml"
        )

    # Active prompt version in changelog
    ver = ec.active_prompt_version
    check(
        f"active_version '{ver}' in changelog",
        "YES" if ver in ec.prompt_versioning.changelog else "NO — missing changelog entry"
    )

    print("\n  ✅  Cross-config consistency: PASSED")
except Exception as exc:
    fail("Cross-config consistency check failed", exc)


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  🎉  ALL CONFIG CHECKS PASSED — ready to build src/")
print("="*60 + "\n")