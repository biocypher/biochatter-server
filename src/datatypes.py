
from typing import Any, List, Optional
from pydantic import BaseModel

class ConnectionArgs(BaseModel):
    host: str
    port: str
    user: Optional[str]=None
    password: Optional[str]=None

class RagConfig(BaseModel):
    splitByChar: bool
    chunkSize: int
    overlapSize: int
    resultNum: int
    connectionArgs: ConnectionArgs
    docIdsWorkspace: Optional[List[str]]=None
    description: Optional[str]=None

class KGConfig(BaseModel):
    resultNum: int
    connectionArgs: ConnectionArgs
    description: Optional[str]=None

class RagNewDocumentPostModel(BaseModel):
    tmpFile: str
    filename: str
    ragConfig: str

class RagAllDocumentsPostModel(BaseModel):
    connectionArgs: ConnectionArgs
    docIds: Optional[List[str]] = None

class RagDocumentDeleteModel(BaseModel):
    connectionArgs: ConnectionArgs
    docId: str
    docIds: Optional[List[str]] = None

class RagConnectionStatusPostModel(BaseModel):
    connectionArgs: ConnectionArgs

class KgConnectionStatusPostModel(BaseModel):
    connectionArgs: ConnectionArgs

class Message(BaseModel):
    role: str
    content: str

class OncoKBConfig(BaseModel):
    useOncoKB: bool
    description: Optional[str] = None

class ChatCompletionsPostModel(BaseModel):
    session_id: str
    messages: List[Message]
    model: str
    temperature: int
    presence_penalty: int
    frequency_penalty: int
    top_p: int
    useRAG: bool
    ragConfig: Optional[RagConfig]=None
    useKG: bool
    kgConfig: Optional[KGConfig]=None
    stream: Optional[bool]=None
    oncokbConfig: Optional[OncoKBConfig]=None
    useAutoAgent: Optional[bool]=None
    
    








