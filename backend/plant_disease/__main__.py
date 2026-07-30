"""python -m plant_disease"""

from plant_disease import config
from plant_disease.app import create_app


def main() -> None:
    application = create_app(load_model=True)
    application.run(host=config.HOST, port=config.PORT, debug=False)


if __name__ == "__main__":
    main()
