from execution import PromptExecutor


def test_executor_built_without_a_manager_leaves_assets_off():
    # Only startup initialises the asset database, so a manager it didn't pass can't be enabled.
    executor = PromptExecutor(object(), cache_type=False, cache_args={"ram": 0, "ram_inactive": 0})

    assert executor.asset_manager.enabled is False
