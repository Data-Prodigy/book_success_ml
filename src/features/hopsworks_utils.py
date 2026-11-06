"""One place that logs in to Hopsworks, so the feature, training, inference and API code all use the
same host / project / engine. engine="python" avoids needing Java/Spark on your machine."""
from config import HOPSWORKS_HOST, HOPSWORKS_PROJECT, require_env


def login():
    import hopsworks  # lazy: the rest of the code (and the unit tests) don't need it installed
    return hopsworks.login(
        host=HOPSWORKS_HOST, port=443, project=HOPSWORKS_PROJECT,
        api_key_value=require_env("HOPSWORKS_API_KEY"), engine="python",
    )