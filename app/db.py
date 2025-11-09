import os
from databricks import sql

def get_conn():
    return sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOST"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_ACCESS_TOKEN"),
    )