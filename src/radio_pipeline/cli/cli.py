import argparse

from src.radio_pipeline.consumer.runner import run_consumer
from src.radio_pipeline.producer.runner import run_producer


def main() -> None:
    parser = argparse.ArgumentParser(description="Radio pipeline runner")
    parser.add_argument(
        "target",
        choices=["producer", "consumer"],
        help="Subsystem to execute",
    )

    args = parser.parse_args()

    if args.target == "producer":
        run_producer()
    elif args.target == "consumer":
        run_consumer()


if __name__ == "__main__":
    main()