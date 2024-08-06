import json
from typing import Optional
from biochatter.rag_agent import RagAgent, RagAgentModeEnum
import logging
import neo4j_utils as nu

logger = logging.getLogger(__name__)

def find_schema_info_node(connection_args: dict):
    try:
        """
        Look for a schema info node in the connected BioCypher graph and load the
        schema info if present.
        """
        db_uri = "bolt://" + connection_args.get("host") + \
            ":" + f"{connection_args.get('port')}"
        neodriver = nu.Driver(
            db_name=connection_args.get("db_name") or "neo4j",
            db_uri=db_uri,
        )
        result = neodriver.query("MATCH (n:Schema_info) RETURN n LIMIT 1")

        if result[0]:
            schema_info_node = result[0][0]["n"]
            schema_dict = json.loads(schema_info_node["schema_info"])
            return schema_dict

        return None
    except Exception as e:
        logger.error(e)
        return None

def get_connection_status(connection_args: Optional[dict]):
    if not connection_args:
        return False
    try:
        schema_dict = find_schema_info_node(connection_args)
        rag_agent = RagAgent(
            mode=RagAgentModeEnum.KG,
            model_name="gpt-3.5-turbo",
            connection_args=connection_args,
            schema_config_or_info_dict=schema_dict,
        )
        return rag_agent.agent.is_connected()
    except Exception as e:
        logger.error(e)
        return False
