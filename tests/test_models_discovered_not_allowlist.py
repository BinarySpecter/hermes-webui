"""Regression coverage for providers with ``models_discovered: true``.

Upstream Hermes Agent treats a ``models`` mapping as per-model metadata, not a
picker allowlist, when the entry carries ``models_discovered: true`` (a catalog
Hermes itself persisted after a successful /v1/models probe). WebUI must do the
same: the live /v1/models catalog stays authoritative and the configured subset
must not clamp the dropdown to its own keys (#7404).
"""


def _provider_group(payload: dict, provider_id: str) -> dict:
    for group in payload.get("groups", []):
        if group.get("provider_id") == provider_id:
            return group
    raise AssertionError(f"provider group {provider_id!r} not found: {payload.get('groups')!r}")


def _ids_from_group(group: dict) -> list[str]:
    return [m["id"] for m in group["models"]]


def _setup(monkeypatch, tmp_path, cfg, live_models):
    import api.config as config

    monkeypatch.setattr(config, "cfg", cfg, raising=False)
    monkeypatch.setattr(config, "_get_config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr(config, "_get_auth_store_path", lambda: tmp_path / "auth.json")
    monkeypatch.setattr(config, "_get_models_cache_path", lambda: tmp_path / "models_cache.json")
    monkeypatch.setattr(config, "_models_cache_source_fingerprint", lambda: {"test": "fingerprint"})
    monkeypatch.setattr(config, "reload_config_if_stale", lambda: None)
    monkeypatch.setattr(config, "reload_config", lambda: None)
    monkeypatch.setattr(config, "_cfg_mtime", 0.0, raising=False)
    monkeypatch.setattr(config, "_LIVE_REBUILD_BUDGET_SECONDS", 0.0, raising=False)
    monkeypatch.setattr(
        config,
        "_read_live_provider_model_ids",
        lambda pid: live_models if pid == cfg["model"]["provider"] else [],
    )
    config.invalidate_models_cache()
    return config


def test_models_discovered_catalog_is_not_a_picker_allowlist(monkeypatch, tmp_path):
    """models_discovered: true must surface the full live catalog, not just the
    configured subset."""
    live_models = [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v4-lite",
        "deepseek-r1-0706",
        "deepseek-v4-reasoner",
    ]
    cfg = {
        "model": {"default": "deepseek-v4-pro", "provider": "deepseek"},
        "providers": {
            "deepseek": {
                "name": "DeepSeek",
                "models_discovered": True,
                "models": {
                    "deepseek-v4-pro": {"context_length": 65536},
                    "deepseek-v4-flash": {"context_length": 32768},
                },
            }
        },
    }
    config = _setup(monkeypatch, tmp_path, cfg, live_models)

    payload = config.get_available_models(force_refresh=True)
    group = _provider_group(payload, "deepseek")
    ids = _ids_from_group(group)

    assert set(ids) == set(live_models), f"expected full live catalog, got {ids}"
    assert "deepseek-v4-lite" in ids
    assert "deepseek-r1-0706" in ids


def test_models_discovered_false_keeps_configured_allowlist(monkeypatch, tmp_path):
    """Without the discovered flag the configured models remain a strict
    allowlist (existing behavior must not regress)."""
    live_models = [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v4-lite",
        "deepseek-r1-0706",
    ]
    cfg = {
        "model": {"default": "deepseek-v4-pro", "provider": "deepseek"},
        "providers": {
            "deepseek": {
                "name": "DeepSeek",
                "models_discovered": False,
                "models": {
                    "deepseek-v4-pro": {"context_length": 65536},
                    "deepseek-v4-flash": {"context_length": 32768},
                },
            }
        },
    }
    config = _setup(monkeypatch, tmp_path, cfg, live_models)

    payload = config.get_available_models(force_refresh=True)
    group = _provider_group(payload, "deepseek")
    ids = _ids_from_group(group)

    assert set(ids) == {"deepseek-v4-pro", "deepseek-v4-flash"}, f"expected configured allowlist, got {ids}"


def test_legacy_discovered_sentinel_is_also_not_an_allowlist(monkeypatch, tmp_path):
    """The legacy in-mapping ``__discovered_model_catalog__`` sentinel written
    by older Hermes versions must behave like ``models_discovered: true``."""
    live_models = [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v4-lite",
        "deepseek-r1-0706",
    ]
    cfg = {
        "model": {"default": "deepseek-v4-pro", "provider": "deepseek"},
        "providers": {
            "deepseek": {
                "name": "DeepSeek",
                "models": {
                    "__discovered_model_catalog__": True,
                    "deepseek-v4-pro": {"context_length": 65536},
                    "deepseek-v4-flash": {"context_length": 32768},
                },
            }
        },
    }
    config = _setup(monkeypatch, tmp_path, cfg, live_models)

    payload = config.get_available_models(force_refresh=True)
    group = _provider_group(payload, "deepseek")
    ids = _ids_from_group(group)

    assert set(ids) == set(live_models), f"expected full live catalog, got {ids}"
    assert "__discovered_model_catalog__" not in ids