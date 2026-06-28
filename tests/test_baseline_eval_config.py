import copy
import unittest

IMPORT_ERROR = None
try:
    from gymnax_exchange.jaxrl.MARL.baseline_eval.baseline_JAXMARL import get_ma_config
except ModuleNotFoundError as exc:
    IMPORT_ERROR = exc
    get_ma_config = None


BASE_CONFIG = {
    "AGENT_CONFIGS": {
        "MarketMaking": {},
        "Execution": {},
    },
    "BASELINE_CONFIGS": {
        "MarketMaking": {"action_space": "AvSt"},
        "Execution": {"action_space": "twap"},
    },
    "dict_of_agents_configs": {
        "MarketMaking": {"action_space": "fixed_quants"},
        "Execution": {"action_space": "fixed_quants_complex"},
    },
    "NUM_AGENTS_PER_TYPE": [1, 1],
    "SEED": 42,
    "EvalTimePeriod": "val",
    "world_config": {
        "alphatradePath": "/tmp/JaxMARL-HFT",
        "dataPath": "/tmp/JaxMARL-HFT/data",
        "stock": "AMZN",
        "timePeriod": "train",
    },
}


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"baseline eval dependencies are not installed: {IMPORT_ERROR.name if IMPORT_ERROR else ''}",
)
class BaselineEvalConfigTest(unittest.TestCase):
    def _action_spaces_for(self, policy_choice, combo_desc):
        ma_config = get_ma_config(copy.deepcopy(BASE_CONFIG), policy_choice, combo_desc)
        self.assertEqual(ma_config.world_config.timePeriod, "val")
        return {
            agent_type: agent_config.action_space
            for agent_type, agent_config in ma_config.dict_of_agents_configs.items()
        }

    def test_policy_combinations_use_expected_learned_and_baseline_configs(self):
        self.assertEqual(
            self._action_spaces_for([0, 0], "BB"),
            {"MarketMaking": "AvSt", "Execution": "twap"},
        )
        self.assertEqual(
            self._action_spaces_for([0, 1], "BL"),
            {"MarketMaking": "AvSt", "Execution": "fixed_quants_complex"},
        )
        self.assertEqual(
            self._action_spaces_for([1, 0], "LB"),
            {"MarketMaking": "fixed_quants", "Execution": "twap"},
        )
        self.assertEqual(
            self._action_spaces_for([1, 1], "LL"),
            {"MarketMaking": "fixed_quants", "Execution": "fixed_quants_complex"},
        )


if __name__ == "__main__":
    unittest.main()
