import src.utils.coordinator_utils as coord_ut
import src.utils.io_utils as io_ut


def main():

    for i in range(1):  # You can change adapted config with overrides inside the loop
        config = io_ut.get_config()
        # run learning
        coord_ut.start_ma_learning(config)


if __name__ == "__main__":
    main()
    print("Done!")
