import os
from datetime import datetime
import logging
from typing import Optional, Dict, List, Any
from biochatter.llm_connect import (
    AzureGptConversation,
    GptConversation,
    WasmConversation,
)
from biochatter.rag_agent import RagAgent, RagAgentModeEnum
from pprint import pprint
import threading
from threading import RLock
from src.constants import (
    ARGS_CONNECTION_ARGS,
    ARGS_DOCIDS_WORKSPACE,
    ARGS_RESULT_NUM,
    AZURE_OPENAI_ENDPOINT,
    OPENAI_API_KEY,
    OPENAI_API_TYPE,
    OPENAI_API_VERSION,
    OPENAI_DEPLOYMENT_NAME,
    OPENAI_MODEL,
)
from src.kg_agent import find_schema_info_node
from src.utils import (
    get_azure_embedding_deployment,
    get_embedding_function,
    get_rag_agent_prompts,
)

logger = logging.getLogger(__name__)

rlock = RLock()

defaultModelConfig = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "sendMemory": True,
    "historyMessageCount": 4,
    "compressMessageLengthThreshold": 2000,
}

MAX_AGE = 3 * 24 * 3600 * 1000  # 3 days


class SessionData:
    def __init__(
        self,
        sessionId: str,
        modelConfig: Dict,
    ):
        self.modelConfig = modelConfig
        self.sessionId = sessionId

        self.createdAt = int(datetime.now().timestamp() * 1000)  # in milliseconds
        self.refreshedAt = self.createdAt
        self.maxAge = MAX_AGE
        self.chatter = self._create_conversation()

    def chat(
        self,
        messages: List[Dict[str, str]],
        authKey: str,
        useRAG: bool = False,
        useKG: bool = False,
        useAutoAgent = False,
        ragConfig: Optional[Dict] = None,
        kgConfig: Optional[Dict] = None,
        oncokbConfig: Optional[Dict] = None,
    ):
        if self.chatter is None:
            return
        if not messages or len(messages) == 0:
            return
        api_key = authKey
        if not isinstance(
            self.chatter, AzureGptConversation
        ):  # chatter is instance of GptConversation
            import openai

            if not openai.api_key or not hasattr(self.chatter, "chat"):
                if not authKey:
                    return False
                # save api_key to os.environ to facilitate conversation_factory
                # to create conversation
                if isinstance(self.chatter, GptConversation):
                    os.environ["OPENAI_API_KEY"] = api_key
                self.chatter.set_api_key(api_key, self.sessionId)

        self._update_rags(
            useRAG=useRAG,
            ragConfig=ragConfig,
            useKG=useKG,
            kgConfig=kgConfig,
            oncokbConfig=oncokbConfig,
            useAutoAgent=useAutoAgent,
        )

        text = messages[-1]["content"]
        messages = messages[:-1]
        # pprint(messages)
        self._setup_messages(messages)
        try:
            (msg, usage, _) = self.chatter.query(text)
            contexts = self.chatter.get_last_injected_context()
            return (msg, usage, contexts)
        except Exception as e:
            logger.error(e)
            raise e

    def _setup_messages(self, openai_msgs: List[Any]):
        if self.chatter is None:
            return False
        self.chatter.messages = []
        for msg in openai_msgs:
            if msg["role"] == "system":
                self.chatter.append_system_message(msg["content"])
            elif msg["role"] == "assistant":
                self.chatter.append_ai_message(msg["content"])
            elif msg["role"] == "user":
                self.chatter.append_user_message(msg["content"])

    def _create_conversation(self):
        if OPENAI_API_TYPE in os.environ and os.environ[OPENAI_API_TYPE] == "azure":
            logger.info("create AzureGptConversation")
            chatter = AzureGptConversation(
                deployment_name=os.environ[OPENAI_DEPLOYMENT_NAME],
                model_name=os.environ[OPENAI_MODEL],
                prompts={"rag_agent_prompts": get_rag_agent_prompts()},
                version=os.environ[OPENAI_API_VERSION],
                base_url=os.environ[AZURE_OPENAI_ENDPOINT],
            )
            chatter.set_api_key(os.environ[OPENAI_API_KEY])
        elif (
            OPENAI_API_TYPE in os.environ
            and os.environ[OPENAI_API_TYPE] == "wasm"
            or self.modelConfig["model"] == "mistral-wasm"
        ):
            logger.info("create WasmConversation")
            chatter = WasmConversation("mistral-wasm", prompts={})
        else:
            logger.info("create GptConversation")
            chatter = GptConversation(
                "gpt-3.5-turbo", prompts={"rag_agent_prompts": get_rag_agent_prompts()}
            )
            temp_api_key = os.environ.get("OPENAI_API_KEY", None)
            if temp_api_key is not None:
                chatter.set_api_key(temp_api_key, self.sessionId)
        return chatter
    
    def _disable_agent(self, agent_mode: str):
        if self.chatter is None:
            return
        _, agent = self.chatter.find_rag_agent(agent_mode)
        if agent is None:
            return
        agent.use_prompt = False
        self.chatter.set_rag_agent(agent)
    def _update_vectorstore_agent(
        self,
        useRAG: bool,
        ragConfig: Optional[Dict]=None,
        useAutoAgent: Optional[bool]=False,
    ):
        if ragConfig is None:
            # disabled
            self._disable_agent(RagAgentModeEnum.VectorStore)            
            return
        # update rag_agent
        try:
            (is_azure, azure_deployment, endpoint) = get_azure_embedding_deployment()
            doc_ids = ragConfig.get(ARGS_DOCIDS_WORKSPACE, None)
            embedding_func = get_embedding_function(
                is_azure=is_azure,
                azure_deployment=azure_deployment,
                azure_endpoint=endpoint,
            )
            rag_agent = RagAgent(
                mode=RagAgentModeEnum.VectorStore,
                model_name=os.environ.get(OPENAI_MODEL, "gpt-3.5-turbo"),
                connection_args=ragConfig[ARGS_CONNECTION_ARGS],
                use_prompt=useRAG or useAutoAgent,
                embedding_func=embedding_func,
                documentids_workspace=doc_ids,
                n_results=ragConfig.get(ARGS_RESULT_NUM, 3),
            )
            if "description" in ragConfig and ragConfig["description"] is not None:
                rag_agent.agent_description = ragConfig["description"]
            self.chatter.set_rag_agent(rag_agent)
        except Exception as e:
            logger.error(e)
    
    def _update_kg_agent(
        self,
        useKG: bool,
        kgConfig: Optional[Dict]=None,
        useAutoAgent: Optional[bool]=False,
    ):
        if kgConfig is None:
            self._disable_agent(RagAgentModeEnum.KG)
            return
        try:
            schema_info = find_schema_info_node(kgConfig["connectionArgs"])
            if schema_info is not None:
                kg_agent = RagAgent(
                    mode=RagAgentModeEnum.KG,
                    model_name=os.environ.get(OPENAI_MODEL, "gpt-3.5-turbo"),
                    connection_args=kgConfig["connectionArgs"],
                    use_prompt=useKG or useAutoAgent,
                    schema_config_or_info_dict=schema_info,
                    conversation_factory=self._create_conversation,
                    n_results=kgConfig.get(ARGS_RESULT_NUM, 3)
                )
                if "description" in kgConfig and kgConfig["description"] is not None:
                    kg_agent.agent_description = kgConfig["description"]
                self.chatter.set_rag_agent(kg_agent)
        except Exception as e:
            logger.error(e)

    def _update_oncokb_agent(
        self,
        oncokbConfig: Optional[Dict]=None,
        useAutoAgent: Optional[bool]=False,
    ):
        if oncokbConfig is None:
            self._disable_agent(RagAgentModeEnum.API_ONCOKB)
            return
        oncokb_agent = RagAgent(
            mode=RagAgentModeEnum.API_ONCOKB,
            conversation_factory=self._create_conversation,
            use_prompt=oncokbConfig["useOncoKB"] or useAutoAgent ,
        )
        if "description" in oncokbConfig:
            oncokb_agent.agent_description = oncokbConfig["description"]
        self.chatter.set_rag_agent(oncokb_agent)

    def _update_rags(
            self, 
            useRAG: bool, 
            useKG: bool, 
            useAutoAgent: bool,
            ragConfig: Optional[Dict]=None, 
            kgConfig: Optional[Dict]=None,
            oncokbConfig: Optional[Dict]=None,
        ):
        self._update_vectorstore_agent(
            useRAG=useRAG, ragConfig=ragConfig, useAutoAgent=useAutoAgent
        )
        self._update_kg_agent(useKG=useKG, kgConfig=kgConfig, useAutoAgent=useAutoAgent)
        self._update_oncokb_agent(oncokbConfig=oncokbConfig, useAutoAgent=useAutoAgent)
        
        self.chatter.use_ragagent_selector = useAutoAgent


conversationsDict = {}


def initialize_conversation(sessionId: str, modelConfig: dict):
    rlock.acquire()
    try:
        conversationsDict[sessionId] = SessionData(
            sessionId=sessionId,
            modelConfig=modelConfig,
        )
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        rlock.release()


def has_conversation(sessionId: str) -> bool:
    rlock.acquire()
    try:
        return sessionId in conversationsDict
    finally:
        rlock.release()


def get_conversation(sessionId: str) -> Optional[SessionData]:
    rlock.acquire()
    try:
        if sessionId not in conversationsDict:
            initialize_conversation(sessionId, defaultModelConfig.copy())
        return conversationsDict[sessionId]
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        rlock.release()


def remove_conversation(sessionId: str):
    rlock.acquire()
    try:
        if sessionId not in conversationsDict:
            return
        del conversationsDict[sessionId]
    except Exception as e:
        logger.error(e)
    finally:
        rlock.release()


def chat(
    sessionId: str,
    messages: List[str],
    authKey: str,
    useRAG: bool,
    useKG: bool,
    useAutoAgent: Optional[bool] = None,
    ragConfig: Optional[Dict]=None,
    kgConfig: Optional[Dict]=None,
    oncokbConfig: Optional[dict] = None,
):
    rlock.acquire()
    useAutoAgent = False if useAutoAgent is None else useAutoAgent
    try:
        conversation = get_conversation(sessionId=sessionId)
        logger.info(
            f"get conversation for session id {sessionId}, "
            "type of conversation is SessionData "
            f"{isinstance(conversation, SessionData)}"
        )
        return conversation.chat(
            messages=messages,
            authKey=authKey,
            ragConfig=ragConfig,
            useRAG=useRAG,
            kgConfig=kgConfig,
            useKG=useKG,
            useAutoAgent=useAutoAgent,
            oncokbConfig=oncokbConfig,
        )
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        rlock.release()


def recycle_conversations():
    logger.info(f"[recycle] - {threading.get_native_id()} recycle_conversation")
    rlock.acquire()
    now = datetime.now().timestamp() * 1000  # in milliseconds
    sessionsToRemove: List[str] = []
    try:
        for sessionId in conversationsDict.keys():
            conversation = get_conversation(sessionId=sessionId)
            assert conversation is not None
            logger.info(
                f"[recycle] sessionId is {sessionId}, "
                f"refreshAt: {conversation.refreshedAt}, "
                f"maxAge: {conversation.maxAge}"
            )
            if conversation.refreshedAt + conversation.maxAge < now:
                sessionsToRemove.append(conversation.sessionId)
        for sessionId in sessionsToRemove:
            remove_conversation(sessionId)
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        rlock.release()
