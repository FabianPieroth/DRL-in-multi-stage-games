import src.utils.coordinator_utils as coord_ut
import src.utils.io_utils as io_ut


def main():
    for i in range(1):  # You can change adapted config with overrides inside the loop
        overrides = [
            f"rl_envs=coin_game",
            f"num_envs=164",
            f"eval_freq={1}",
            f"device={6}",
        ]
        config = io_ut.get_config(overrides=overrides)
        # run learning
        coord_ut.start_ma_learning(config)


if __name__ == "__main__":
    main()
    print("Done!")
