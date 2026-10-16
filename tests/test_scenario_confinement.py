"""Security tests for signed scenario capabilities and one-shot confinement.

These tests cover the properties that file checksums cannot provide:
tamper-evident capability signatures, the narrow worker IPC contract,
fail-closed audit consistency, and (where a kernel backend exists) the
actual network/filesystem/environment guarantees. Best-effort execution is
exercised separately and must never be described as a security boundary.
"""

import hashlib
import importlib.util
import json
import os
import secrets
import shutil
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "codex-instruct.py"
SCENARIO_ROOT = Path(__file__).resolve().parents[1] / "scenarios"
spec = importlib.util.spec_from_file_location("codex_instruct_confinement", MODULE_PATH)
codex_instruct = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = codex_instruct
spec.loader.exec_module(codex_instruct)

STRONG_BACKEND = codex_instruct._scenario_detect_strong_backend("auto")
requires_strong_backend = pytest.mark.skipif(
    STRONG_BACKEND is None,
    reason="no kernel confinement backend (sandbox-exec/bwrap/docker/podman)",
)


def _copy_example(tmp_path):
    root = (tmp_path / "scenario library").resolve()
    shutil.copytree(SCENARIO_ROOT, root)
    return root, root / "example_fixture"


def _load(package):
    return json.loads((package / "scenario.json").read_text(encoding="utf-8"))


def _store(package, data):
    (package / "scenario.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resign(package, data, key=None):
    data["checksums"] = {
        relative: hashlib.sha256((package / relative).read_bytes()).hexdigest()
        for relative in data["checksums"]
    }
    signing_key = key or codex_instruct._scenario_first_party_signing_key()
    data["security"]["signature"] = (
        codex_instruct._scenario_compute_capability_signature(data, signing_key)
    )
    _store(package, data)


# ─── capability signatures ──────────────────────────────────────────────────


def test_package_without_security_block_is_rejected(tmp_path):
    root, package = _copy_example(tmp_path)
    data = _load(package)
    del data["security"]
    _store(package, data)

    with pytest.raises(ValueError, match="root fields are invalid"):
        codex_instruct.load_scenario_package(root, "example_fixture")


def test_package_with_forged_signature_is_rejected(tmp_path):
    root, package = _copy_example(tmp_path)
    data = _load(package)
    data["security"]["signature"] = (
        codex_instruct.SCENARIO_SIGNATURE_PREFIX + "0" * 64
    )
    _store(package, data)

    with pytest.raises(codex_instruct.HooksConflict, match="signature is invalid or untrusted"):
        codex_instruct.load_scenario_package(root, "example_fixture")


def test_tampering_with_a_signed_capability_invalidates_the_signature(tmp_path):
    root, package = _copy_example(tmp_path)
    data = _load(package)
    data["security"]["capabilities"]["network"] = True
    _store(package, data)

    with pytest.raises(codex_instruct.HooksConflict, match="signature is invalid or untrusted"):
        codex_instruct.load_scenario_package(root, "example_fixture")


def test_tampering_with_a_signed_entrypoint_invalidates_the_signature(tmp_path):
    root, package = _copy_example(tmp_path)
    data = _load(package)
    data["requires"] = [
        {"name": "x", "type": "command", "version": ">=1", "probe": ["x", "--version"]}
    ]
    _store(package, data)

    with pytest.raises(codex_instruct.HooksConflict, match="signature is invalid or untrusted"):
        codex_instruct.load_scenario_package(root, "example_fixture")


def test_operator_pinned_key_replaces_the_first_party_key(tmp_path, monkeypatch):
    root, package = _copy_example(tmp_path)

    # A host pinned to an operator key must reject release first-party
    # signatures outright (no trust union, no downgrade).
    operator_key = secrets.token_bytes(32)
    monkeypatch.setenv(
        codex_instruct.SCENARIO_TRUSTED_KEY_ENV, operator_key.hex()
    )

    with pytest.raises(codex_instruct.HooksConflict, match="signature is invalid or untrusted"):
        codex_instruct.load_scenario_package(root, "example_fixture")

    # Re-signing with the pinned key restores the trusted path.
    data = _load(package)
    _resign(package, data, key=operator_key)
    loaded = codex_instruct.load_scenario_package(root, "example_fixture")
    assert loaded.security.signature_valid is True
    assert loaded.security.signature_key_source == "operator-pinned"


def test_loaded_package_carries_the_authenticated_capability_set():
    package = codex_instruct.load_scenario_package(SCENARIO_ROOT, "example_fixture")
    assert package.security.isolation == "strong"
    assert package.security.signature_valid is True
    assert package.security.signature_key_source == "release-pinned-first-party"
    assert package.security.capabilities["network"] is False
    assert package.security.capabilities["filesystem"] == "read-only"


# ─── narrow worker IPC ───────────────────────────────────────────────────────


def _caps(**overrides):
    capabilities = dict(codex_instruct.SCENARIO_DEFAULT_CAPABILITIES)
    capabilities["max_output_bytes"] = 1024
    capabilities.update(overrides)
    return codex_instruct.ScenarioConfinementCaps.from_manifest(capabilities)


def _envelope(**overrides):
    envelope = {
        "schema": codex_instruct.SCENARIO_WORKER_RESULT_SCHEMA,
        "exit_code": 0,
        "timed_out": False,
        "duration_seconds": 0.01,
        "peak_rss_kb": 0,
        "stdout": "ok\n",
        "stdout_truncated": False,
        "stderr": "",
        "stderr_truncated": False,
        "limits_applied": {},
        "policy_digest": "a" * 64,
        "worker_error": None,
    }
    envelope.update(overrides)
    return envelope


def _raw(envelope):
    return json.dumps(envelope, sort_keys=True).encode("utf-8")


def test_worker_envelope_accepts_a_strict_valid_envelope():
    parsed = codex_instruct._scenario_parse_worker_envelope(
        _raw(_envelope()), "a" * 64, _caps()
    )
    assert parsed["exit_code"] == 0
    assert parsed["stdout"] == "ok\n"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.update(schema="evil/v1"),
        lambda e: e.update(policy_digest="b" * 64),
        lambda e: e.update(extra=True),
        lambda e: e.pop("worker_error"),
        lambda e: e.update(exit_code="0"),
        lambda e: e.update(stdout="x" * 1025),
    ],
)
def test_worker_envelope_rejects_tampered_or_oversized_contracts(mutate):
    envelope = _envelope()
    mutate(envelope)
    with pytest.raises(RuntimeError):
        codex_instruct._scenario_parse_worker_envelope(
            _raw(envelope), "a" * 64, _caps()
        )


def test_worker_envelope_rejects_non_json_and_oversized_bytes():
    with pytest.raises(RuntimeError, match="malformed"):
        codex_instruct._scenario_parse_worker_envelope(b"not json", "a" * 64, _caps())
    with pytest.raises(RuntimeError, match="exceeds"):
        codex_instruct._scenario_parse_worker_envelope(
            b"x" * (2 * 1024 + 64 * 1024 + 1), "a" * 64, _caps()
        )


# ─── audit consistency (fail-closed self-check) ─────────────────────────────


def _strong_audit(**overrides):
    audit = {
        "enforced_mode": "strong",
        "guarantees": {
            "network_denied": True,
            "writes_confined_to_scratch": True,
            "home_credentials_denied": True,
            "environment_minimized": True,
            "cpu_limited": True,
            "memory_limited": False,
            "processes_limited": False,
            "wall_timeout_enforced": True,
            "os_kernel_enforced": True,
        },
        "warnings": [],
    }
    audit.update(overrides)
    return audit


def test_strong_audit_without_kernel_enforcement_is_rejected():
    audit = _strong_audit()
    audit["guarantees"]["os_kernel_enforced"] = False
    with pytest.raises(RuntimeError, match="kernel enforcement"):
        codex_instruct._scenario_require_audit_consistency(audit, _caps())


def test_strong_audit_must_match_the_signed_network_grant():
    audit = _strong_audit()
    audit["guarantees"]["network_denied"] = False
    with pytest.raises(RuntimeError, match="network grant"):
        codex_instruct._scenario_require_audit_consistency(audit, _caps())


def test_best_effort_audit_must_not_claim_isolation_guarantees():
    base = {
        "enforced_mode": "best-effort",
        "guarantees": {
            "network_denied": False,
            "writes_confined_to_scratch": False,
            "home_credentials_denied": False,
            "environment_minimized": True,
            "cpu_limited": False,
            "memory_limited": False,
            "processes_limited": False,
            "wall_timeout_enforced": True,
            "os_kernel_enforced": False,
        },
        "warnings": [codex_instruct.SCENARIO_BEST_EFFORT_WARNING],
    }
    codex_instruct._scenario_require_audit_consistency(base, _caps())

    lying = json.loads(json.dumps(base))
    lying["guarantees"]["network_denied"] = True
    with pytest.raises(RuntimeError, match="must not claim"):
        codex_instruct._scenario_require_audit_consistency(lying, _caps())

    kernel_lying = json.loads(json.dumps(base))
    kernel_lying["guarantees"]["os_kernel_enforced"] = True
    with pytest.raises(RuntimeError, match="kernel enforcement"):
        codex_instruct._scenario_require_audit_consistency(kernel_lying, _caps())

    silent = json.loads(json.dumps(base))
    silent["warnings"] = []
    with pytest.raises(RuntimeError, match="explicit warning"):
        codex_instruct._scenario_require_audit_consistency(silent, _caps())


def test_best_effort_audit_uses_no_secure_or_sandbox_wording():
    base = {
        "enforced_mode": "best-effort",
        "guarantees": {
            "network_denied": False,
            "writes_confined_to_scratch": False,
            "home_credentials_denied": False,
            "environment_minimized": True,
            "cpu_limited": False,
            "memory_limited": False,
            "processes_limited": False,
            "wall_timeout_enforced": True,
            "os_kernel_enforced": False,
        },
        "warnings": [codex_instruct.SCENARIO_BEST_EFFORT_WARNING],
        "backend": "process-rlimits",
    }
    rendered = json.dumps(base).lower()
    assert "sandbox" not in rendered
    assert "secure" not in rendered


# ─── readiness and policy gating ────────────────────────────────────────────


def test_strong_scenario_without_backend_reports_a_blocker(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_instruct, "_scenario_detect_strong_backend", lambda _preferred: None)
    security = codex_instruct.ScenarioSecurity(
        isolation="strong",
        capabilities=dict(codex_instruct.SCENARIO_DEFAULT_CAPABILITIES),
        signature=codex_instruct.SCENARIO_SIGNATURE_PREFIX + "0" * 64,
        signature_valid=True,
        signature_key_source="release-pinned-first-party",
    )
    blocker = codex_instruct.scenario_isolation_ready(security)
    assert blocker is not None
    assert "no OS confinement backend" in blocker

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    with pytest.raises(codex_instruct.ScenarioConfinementUnavailable):
        codex_instruct.run_scenario_confined(
            [sys.executable, "-c", "print('must not run')"],
            cwd=cwd,
            capabilities=codex_instruct.SCENARIO_DEFAULT_CAPABILITIES,
            declared_isolation="strong",
            purpose="readiness-test",
        )


def test_strong_only_policy_refuses_best_effort_scenarios(monkeypatch, tmp_path):
    monkeypatch.setenv(codex_instruct.SCENARIO_ISOLATION_POLICY_ENV, "strong-only")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    with pytest.raises(codex_instruct.ScenarioConfinementUnavailable, match="requires strong"):
        codex_instruct.run_scenario_confined(
            [sys.executable, "-c", "print('x')"],
            cwd=cwd,
            capabilities=codex_instruct.SCENARIO_DEFAULT_CAPABILITIES,
            declared_isolation="best-effort",
            purpose="policy-test",
        )


# ─── live confinement guarantees ────────────────────────────────────────────


def _workspace(tmp_path):
    cwd = tmp_path / "cwd"
    scratch = tmp_path / "scratch"
    canary_dir = tmp_path / "canary"
    cwd.mkdir(exist_ok=True)
    scratch.mkdir(exist_ok=True)
    canary_dir.mkdir(exist_ok=True)
    (canary_dir / "secret.txt").write_text("credential-canary\n", encoding="utf-8")
    return cwd, scratch, canary_dir


def _run_python(code, tmp_path, *, capabilities=None, isolation="strong", env=None):
    cwd, scratch, _canary = _workspace(tmp_path)
    overridden_env = dict(os.environ)
    if env:
        overridden_env.update(env)
    old_env = dict(os.environ)
    os.environ.update({key: value for key, value in overridden_env.items()})
    try:
        return codex_instruct.run_scenario_confined(
            [sys.executable, "-I", "-B", "-c", code],
            cwd=cwd,
            capabilities=capabilities or codex_instruct.SCENARIO_DEFAULT_CAPABILITIES,
            declared_isolation=isolation,
            purpose="live-security-test",
            scratch_dir=scratch,
        )
    finally:
        os.environ.clear()
        os.environ.update(old_env)


@requires_strong_backend
def test_strong_worker_denies_network(tmp_path):
    result = _run_python(
        "import socket\n"
        "s = socket.socket()\n"
        "s.settimeout(3)\n"
        "try:\n"
        "    s.connect(('1.1.1.1', 53))\n"
        "    print('ALLOWED')\n"
        "except OSError:\n"
        "    print('DENIED')\n",
        tmp_path,
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "DENIED"
    assert result.audit["guarantees"]["network_denied"] is True
    assert result.audit["guarantees"]["os_kernel_enforced"] is True


@requires_strong_backend
def test_strong_worker_denies_reads_outside_carve_roots(tmp_path):
    _cwd, _scratch, canary_dir = _workspace(tmp_path)
    canary = canary_dir / "secret.txt"
    result = _run_python(
        "import pathlib\n"
        f"p = pathlib.Path({str(canary)!r})\n"
        "try:\n"
        "    p.read_text(encoding='utf-8')\n"
        "    print('ALLOWED')\n"
        "except OSError:\n"
        "    print('DENIED')\n",
        tmp_path,
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "DENIED"


@requires_strong_backend
def test_strong_worker_allows_scratch_writes_and_cwd_reads(tmp_path):
    (_cwd, scratch, _canary) = _workspace(tmp_path)
    result = _run_python(
        "import os, pathlib\n"
        "scratch = pathlib.Path(os.environ['TMPDIR'])\n"
        "try:\n"
        "    (scratch / 'probe.txt').write_text('ok', encoding='utf-8')\n"
        "    write = 'ALLOWED'\n"
        "except OSError:\n"
        "    write = 'DENIED'\n"
        "print(write)\n",
        tmp_path,
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "ALLOWED"
    assert (scratch / "probe.txt").read_text(encoding="utf-8") == "ok"
    assert result.audit["guarantees"]["writes_confined_to_scratch"] is True


@requires_strong_backend
def test_strong_worker_strips_credentials_from_the_environment(tmp_path):
    result = _run_python(
        "import os\nprint(os.environ.get('OPENAI_API_KEY'))",
        tmp_path,
        env={"OPENAI_API_KEY": "sk-live-canary-credential"},
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "None"
    assert "sk-live-canary-credential" not in result.stdout + result.stderr
    assert result.audit["guarantees"]["environment_minimized"] is True


@requires_strong_backend
def test_strong_worker_enforces_the_wall_timeout(tmp_path):
    capabilities = dict(codex_instruct.SCENARIO_DEFAULT_CAPABILITIES)
    capabilities["max_wall_seconds"] = 1
    result = _run_python(
        "import time\ntime.sleep(30)\n",
        tmp_path,
        capabilities=capabilities,
    )
    assert result.timed_out is True
    assert result.exit_code != 0
    assert result.duration_seconds < 10
    assert result.audit["guarantees"]["wall_timeout_enforced"] is True


@requires_strong_backend
def test_strong_worker_bounds_output(tmp_path):
    capabilities = dict(codex_instruct.SCENARIO_DEFAULT_CAPABILITIES)
    capabilities["max_output_bytes"] = 1024
    result = _run_python(
        "import sys\nsys.stdout.write('x' * 8192)\nsys.stdout.flush()\n",
        tmp_path,
        capabilities=capabilities,
    )
    assert len(result.stdout) <= 1024
    assert len(result.stderr) <= 1024


# ─── best-effort is honestly labelled and is not a boundary ─────────────────


def test_best_effort_worker_runs_unconfined_and_labels_itself_honestly(tmp_path):
    _cwd, _scratch, canary_dir = _workspace(tmp_path)
    canary = canary_dir / "secret.txt"
    result = _run_python(
        "import pathlib\n"
        f"print(pathlib.Path({str(canary)!r}).read_text(encoding='utf-8').strip())\n",
        tmp_path,
        isolation="best-effort",
    )
    assert result.exit_code == 0, result.stderr
    # The canary is readable: proof that best-effort mode is not isolation.
    assert result.stdout.strip() == "credential-canary"
    assert result.audit["enforced_mode"] == "best-effort"
    guarantees = result.audit["guarantees"]
    assert guarantees["os_kernel_enforced"] is False
    assert guarantees["network_denied"] is False
    assert guarantees["home_credentials_denied"] is False
    assert guarantees["writes_confined_to_scratch"] is False
    assert any(
        "not a security boundary" in warning for warning in result.audit["warnings"]
    )
    rendered = json.dumps(result.audit).lower()
    assert "sandbox" not in rendered
    assert "secure" not in rendered


# ─── maintenance signing helper and key stores ──────────────────────────────


def test_sign_scenario_library_capabilities_refreshes_every_package(tmp_path):
    root, package = _copy_example(tmp_path)
    data = _load(package)
    data["security"]["signature"] = (
        codex_instruct.SCENARIO_SIGNATURE_PREFIX + "0" * 64
    )
    _store(package, data)
    with pytest.raises(codex_instruct.HooksConflict, match="signature is invalid"):
        codex_instruct.load_scenario_package(root, "example_fixture")

    signed = codex_instruct.sign_scenario_library_capabilities(root)

    assert signed == sorted(signed)
    assert "example_fixture" in signed
    loaded = codex_instruct.load_scenario_package(root, "example_fixture")
    assert loaded.security.signature_valid is True
    assert loaded.security.signature_key_source == "release-pinned-first-party"


def test_signing_helper_rejects_a_library_without_packages(tmp_path):
    empty = tmp_path / "empty-library"
    empty.mkdir()
    with pytest.raises(codex_instruct.HooksConflict, match="no scenario packages"):
        codex_instruct.sign_scenario_library_capabilities(empty)


def test_operator_key_file_pins_trust(tmp_path, monkeypatch):
    root, package = _copy_example(tmp_path)
    operator_key = secrets.token_bytes(32)
    key_file = tmp_path / "operator-key.hex"
    key_file.write_text(operator_key.hex() + "\n", encoding="utf-8")
    monkeypatch.setenv(codex_instruct.SCENARIO_SIGNING_KEY_FILE_ENV, str(key_file))
    monkeypatch.setenv(codex_instruct.SCENARIO_TRUSTED_KEY_FILE_ENV, str(key_file))

    assert codex_instruct._scenario_first_party_signing_key() == operator_key
    data = _load(package)
    data["security"]["signature"] = (
        codex_instruct._scenario_compute_capability_signature(data, operator_key)
    )
    _store(package, data)

    loaded = codex_instruct.load_scenario_package(root, "example_fixture")
    assert loaded.security.signature_valid is True
    assert loaded.security.signature_key_source == "operator-pinned-file"
    assert codex_instruct._scenario_trusted_signing_keys()[0][0] == "operator-pinned-file"


def test_invalid_operator_key_hex_is_rejected(monkeypatch):
    monkeypatch.setenv(codex_instruct.SCENARIO_TRUSTED_KEY_ENV, "not-hex")
    with pytest.raises(ValueError, match="64 hex characters"):
        codex_instruct._scenario_trusted_signing_keys()


def test_missing_trusted_key_file_is_a_hard_error(tmp_path, monkeypatch):
    monkeypatch.setenv(
        codex_instruct.SCENARIO_TRUSTED_KEY_FILE_ENV,
        str(tmp_path / "absent.hex"),
    )
    with pytest.raises(RuntimeError, match="cannot read trusted scenario key file"):
        codex_instruct._scenario_trusted_signing_keys()


def test_strong_only_policy_readiness_blocks_best_effort(monkeypatch):
    monkeypatch.setenv(codex_instruct.SCENARIO_ISOLATION_POLICY_ENV, "strong-only")
    security = codex_instruct.ScenarioSecurity(
        isolation="best-effort",
        capabilities=dict(codex_instruct.SCENARIO_DEFAULT_CAPABILITIES),
        signature=codex_instruct.SCENARIO_SIGNATURE_PREFIX + "0" * 64,
        signature_valid=True,
        signature_key_source="release-pinned-first-party",
    )
    blocker = codex_instruct.scenario_isolation_ready(security)
    assert blocker is not None
    assert "not a security boundary" in blocker


# ─── probe failure mapping (no backend needed) ──────────────────────────────


def _probe_requirement(kind):
    if kind == "python-module":
        return {
            "name": "missing_mod_xyz",
            "type": "python-module",
            "version": ">=1",
            "probe": ["-c", "import missing_mod_xyz"],
        }
    return {
        "name": "codex-keysmith-missing-tool",
        "type": "command",
        "version": ">=1",
        "probe": ["codex-keysmith-missing-tool", "--version"],
    }


def test_probe_maps_missing_backend(monkeypatch):
    def refuse(*args, **kwargs):
        raise codex_instruct.ScenarioConfinementUnavailable("no backend")

    monkeypatch.setattr(codex_instruct, "run_scenario_confined", refuse)
    blocker = codex_instruct._scenario_probe_requirement(_probe_requirement("command"))
    assert "probe refused because strong isolation is unavailable" in blocker


def test_probe_maps_missing_command_binary(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(codex_instruct, "run_scenario_confined", missing)
    blocker = codex_instruct._scenario_probe_requirement(_probe_requirement("command"))
    assert "is not on PATH" in blocker


def test_probe_maps_missing_python_module_binary(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(codex_instruct, "run_scenario_confined", missing)
    blocker = codex_instruct._scenario_probe_requirement(
        _probe_requirement("python-module")
    )
    assert "is missing; install python-module" in blocker


def test_probe_maps_worker_file_not_found(monkeypatch):
    result = codex_instruct.ScenarioConfinementResult(
        exit_code=1,
        stdout="",
        stderr="",
        timed_out=False,
        duration_seconds=0.1,
        worker_error="FileNotFoundError: [Errno 2] tool not found",
        audit=None,
    )
    monkeypatch.setattr(
        codex_instruct, "run_scenario_confined", lambda *args, **kwargs: result
    )
    blocker = codex_instruct._scenario_probe_requirement(_probe_requirement("command"))
    assert "is not on PATH" in blocker

    python_blocker = codex_instruct._scenario_probe_requirement(
        _probe_requirement("python-module")
    )
    assert "is missing; install python-module" in python_blocker


def test_probe_maps_generic_worker_error(monkeypatch):
    result = codex_instruct.ScenarioConfinementResult(
        exit_code=1,
        stdout="",
        stderr="boom",
        timed_out=False,
        duration_seconds=0.1,
        worker_error="RuntimeError: boom",
        audit=None,
    )
    monkeypatch.setattr(
        codex_instruct, "run_scenario_confined", lambda *args, **kwargs: result
    )
    blocker = codex_instruct._scenario_probe_requirement(_probe_requirement("command"))
    assert "probe worker failed" in blocker


def test_load_rejects_invalid_scenario_json(tmp_path):
    root, package = _copy_example(tmp_path)
    (package / "scenario.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="scenario.json is invalid"):
        codex_instruct.load_scenario_package(root, "example_fixture")


def test_sign_helper_rejects_id_mismatch_and_invalid_json(tmp_path):
    root, package = _copy_example(tmp_path)
    data = _load(package)
    data["id"] = "different_id"
    _store(package, data)
    with pytest.raises(codex_instruct.HooksConflict, match="does not match its package"):
        codex_instruct.sign_scenario_library_capabilities(root)

    (package / "scenario.json").write_text("garbage", encoding="utf-8")
    with pytest.raises(ValueError, match="scenario.json is invalid"):
        codex_instruct.sign_scenario_library_capabilities(root)


def test_probe_refused_when_frozen_sidecar_has_no_interpreter(monkeypatch):
    def no_interpreter(requirement):
        raise FileNotFoundError("frozen sidecar has no Python interpreter")

    monkeypatch.setattr(codex_instruct, "_scenario_probe_command", no_interpreter)
    blocker = codex_instruct._scenario_probe_requirement(
        _probe_requirement("python-module")
    )
    assert "probe refused because strong isolation is unavailable" in blocker
    assert "frozen sidecar has no Python interpreter" in blocker


def test_seatbelt_profile_builder_is_deny_default_with_grant_carve_outs(tmp_path):
    cwd, scratch, roots, _environment, caps = _builder_inputs(tmp_path)
    profile = codex_instruct._scenario_build_seatbelt_profile(
        scratch, [cwd, *roots], caps
    )
    assert "(deny default)" in profile
    assert "(deny network*)" in profile
    assert "(allow file-write*" in profile
    assert "(deny process-exec" in profile

    writable_caps = dict(codex_instruct.SCENARIO_DEFAULT_CAPABILITIES)
    writable_caps["network"] = True
    writable_caps["filesystem"] = codex_instruct.SCENARIO_FILESYSTEM_SCRATCH
    loose_caps = codex_instruct.ScenarioConfinementCaps.from_manifest(writable_caps)
    loose_profile = codex_instruct._scenario_build_seatbelt_profile(
        scratch, [cwd], loose_caps
    )
    assert "(allow network*)" in loose_profile
    assert "(deny process-exec" not in loose_profile


@pytest.mark.skipif(os.name == "nt", reason="backend probing uses POSIX shims")
def test_backend_detection_probes_explicit_engines(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    podman = fake_bin / "podman"
    podman.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    podman.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    codex_instruct._SCENARIO_BACKEND_CACHE.clear()

    selected = codex_instruct._scenario_detect_strong_backend("podman")
    assert selected == {"name": "podman", "path": str(podman.resolve())}

    bwrap_broken = fake_bin / "bwrap"
    bwrap_broken.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    bwrap_broken.chmod(0o755)
    codex_instruct._SCENARIO_BACKEND_CACHE.clear()
    assert codex_instruct._scenario_detect_strong_backend("bwrap") is None

    bwrap_broken.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex_instruct._SCENARIO_BACKEND_CACHE.clear()
    selected = codex_instruct._scenario_detect_strong_backend("bwrap")
    assert selected is not None
    assert selected["name"] == "bwrap"
    codex_instruct._SCENARIO_BACKEND_CACHE.clear()


# ─── launcher argv builders and process teardown (host independent) ─────────


def _builder_inputs(tmp_path, network=False):
    cwd = tmp_path / "cwd"
    scratch = tmp_path / "scratch"
    carve = tmp_path / "input"
    for path in (cwd, scratch, carve):
        path.mkdir()
    capabilities = dict(codex_instruct.SCENARIO_DEFAULT_CAPABILITIES)
    capabilities["network"] = network
    caps = codex_instruct.ScenarioConfinementCaps.from_manifest(capabilities)
    environment = codex_instruct._scenario_confined_environment(scratch)
    return cwd, scratch, [carve], environment, caps


@pytest.mark.skipif(os.name == "nt", reason="bubblewrap argv uses POSIX ids")
def test_bwrap_argv_denies_network_and_binds_scratch_writable(tmp_path):
    cwd, scratch, roots, environment, caps = _builder_inputs(tmp_path)
    worker_argv = [sys.executable, "-I", "-B", str(scratch / "worker.py")]
    argv = codex_instruct._scenario_build_bwrap_argv(
        "/usr/bin/bwrap", scratch, roots, environment, cwd, caps, worker_argv
    )
    assert "--unshare-net" in argv
    assert "--die-with-parent" in argv and "--new-session" in argv
    bind_index = argv.index("--bind")
    assert argv[bind_index: bind_index + 3] == ["--bind", str(scratch), str(scratch)]
    ro_bind = argv[argv.index("--ro-bind"): argv.index("--ro-bind") + 3]
    assert ro_bind == ["--ro-bind", "/", "/"]
    chdir_index = argv.index("--chdir")
    assert argv[chdir_index: chdir_index + 2] == ["--chdir", str(cwd)]
    assert argv[-len(worker_argv):] == worker_argv
    assert "--clearenv" in argv


@pytest.mark.skipif(os.name == "nt", reason="bubblewrap argv uses POSIX ids")
def test_bwrap_argv_omits_unshare_net_when_network_granted(tmp_path):
    cwd, scratch, roots, environment, caps = _builder_inputs(tmp_path, network=True)
    argv = codex_instruct._scenario_build_bwrap_argv(
        "/usr/bin/bwrap", scratch, roots, environment, cwd, caps, ["python3"]
    )
    assert "--unshare-net" not in argv


@pytest.mark.skipif(os.name == "nt", reason="container argv uses POSIX ids")
def test_container_argv_is_read_only_networkless_and_resource_limited(tmp_path, monkeypatch):
    monkeypatch.setenv(codex_instruct.SCENARIO_SANDBOX_IMAGE_ENV, "registry.example/ks:1")
    cwd, scratch, roots, environment, caps = _builder_inputs(tmp_path)
    argv, plan = codex_instruct._scenario_build_container_argv(
        "docker", scratch, roots, environment, cwd, caps
    )
    assert argv[:3] == ["docker", "run", "--rm"]
    network_index = argv.index("--network")
    assert argv[network_index: network_index + 2] == ["--network", "none"]
    assert "--read-only" in argv
    assert "--cap-drop" in argv and "ALL" in argv
    memory_index = argv.index("--memory")
    assert argv[memory_index: memory_index + 2] == [
        "--memory",
        f"{caps.max_memory_bytes // (1024 * 1024)}m",
    ]
    pids_index = argv.index("--pids-limit")
    assert argv[pids_index: pids_index + 2] == ["--pids-limit", str(caps.max_processes)]
    assert any(f"{scratch}:/worker:rw" in part for part in argv)
    assert plan["image"] == "registry.example/ks:1"
    assert plan["container_scratch"] == "/worker"
    assert plan["container_roots"][0]["container"] == "/input/0"
    # cwd is outside the read-only carve root, so the worker workdir falls
    # back to the writable scratch mount.
    assert plan["cwd_container"] == "/worker"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group teardown")
def test_terminate_confined_process_falls_back_to_pid_kill():
    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid
            self.killed = False

        def kill(self):
            self.killed = True

    process = FakeProcess(999999)
    codex_instruct._scenario_terminate_confined_process(process)
    assert process.killed is True


_FAKE_ENGINE = """\
import json
import os
import sys

args = sys.argv[1:]
scratch = ""
for index, arg in enumerate(args):
    if arg == "--volume":
        host, _, container_spec = args[index + 1].partition(":")
        if container_spec.startswith("/worker"):
            scratch = host
request_path = os.path.join(scratch, "request.json")
host_request_path = os.path.join(scratch, "request.host.json")
with open(request_path, encoding="utf-8") as handle:
    request = json.load(handle)
request["result_path"] = os.path.join(scratch, "result.json")
request["cwd"] = scratch
with open(host_request_path, "w", encoding="utf-8") as handle:
    json.dump(request, handle)
command = ["python3", "-I", "-B"]
command.append(os.path.join(scratch, "worker.py"))
command.extend(["--request", host_request_path])
os.execvp(command[0], command)
"""


@pytest.mark.skipif(os.name == "nt", reason="fake engine uses a POSIX PATH shim")
def test_container_engine_path_runs_worker_and_reports_engine_audit(
    tmp_path, monkeypatch
):
    cwd = tmp_path / "cwd"
    scratch = tmp_path / "scratch"
    cwd.mkdir()
    scratch.mkdir()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    engine = fake_bin / "docker"
    engine.write_text(_FAKE_ENGINE, encoding="utf-8")
    engine.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv(codex_instruct.SCENARIO_BACKEND_ENV, "docker")
    codex_instruct._SCENARIO_BACKEND_CACHE.clear()

    # Inside an image the interpreter is "python3" from PATH; the launcher's
    # host->container argv rewrite leaves that token untouched.
    result = codex_instruct.run_scenario_confined(
        ["python3", "-I", "-B", "-c", "print('fake-engine-ok')"],
        cwd=cwd,
        capabilities=codex_instruct.SCENARIO_DEFAULT_CAPABILITIES,
        declared_isolation="strong",
        purpose="engine-path-test",
        scratch_dir=scratch,
    )

    assert result.exit_code == 0, result.stderr
    assert "fake-engine-ok" in result.stdout
    assert result.audit["backend"] == "docker"
    assert result.audit["enforced_mode"] == "strong"
    assert result.audit["guarantees"]["os_kernel_enforced"] is True
