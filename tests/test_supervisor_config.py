from pathlib import Path

from scalper_hft.live.supervisor_config import SupervisorConfig
from scalper_hft.strategies.regime_supervisor import RegimeSupervisor


def test_supervisor_config_parsing(tmp_path: Path):
    yaml_content = """
name: "Test Supervisor"
max_total_leverage: 1.5
risk_regime_filtering: false

families:
  - name: "pairs_arb"
    allocation: 0.6

strategies:
  - id: "pairs_arb_btcusdt"
    family: "pairs_arb"
    preferred_regimes: ["low_vol", "mean_reverting"]
    allocation_weight: 1.0
    params:
      window: 100
      z_entry: 2.0
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    cfg = SupervisorConfig.from_yaml(config_file)
    assert cfg.name == "Test Supervisor"
    assert cfg.max_total_leverage == 1.5
    assert not cfg.risk_regime_filtering
    
    assert len(cfg.families) == 1
    assert cfg.families[0].name == "pairs_arb"
    assert cfg.families[0].allocation == 0.6
    
    assert len(cfg.strategies) == 1
    assert cfg.strategies[0].id == "pairs_arb_btcusdt"
    assert cfg.strategies[0].family == "pairs_arb"
    assert cfg.strategies[0].params["window"] == 100

def test_regime_supervisor_from_config(tmp_path: Path):
    yaml_content = """
name: "Test Supervisor"
strategies:
  - id: "mean_reversion_1"
    family: "mean_reversion"
    preferred_regimes: ["test_regime"]
    params:
      bb_period: 20
"""
    config_file = tmp_path / "test_config2.yaml"
    config_file.write_text(yaml_content)

    # test invalid/valid instantiation
    sup = RegimeSupervisor.from_config(str(config_file))
    assert sup.name == "regime_supervisor"
    assert "mean_reversion_1" in sup._strat_names
    assert len(sup._strats) == 1
    assert list(sup._strats[0].preferred_regimes)[0] == "test_regime"
